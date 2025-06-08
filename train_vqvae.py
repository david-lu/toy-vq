import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

# Import the VQ-VAE model and other custom modules
from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist


def train(epoch, loader, model, optimizer, scheduler, device):
    """
    ## Single Epoch Training Function
    This function defines the training loop for a single epoch. It iterates over
    the dataset, performs the forward and backward passes, updates the model's
    weights, and logs training progress. It also handles logic for distributed
    training and periodically saves sample image reconstructions to monitor
    the model's performance visually.
    """
    # If this is the primary process in a distributed setup, wrap the loader in tqdm for a progress bar.
    if dist.is_primary():
        loader = tqdm(loader)

    # Mean Squared Error loss is used for the reconstruction loss.
    criterion = nn.MSELoss()

    # Weighting factor for the latent loss (commitment loss) from the VQ layer.
    latent_loss_weight = 0.25
    # Number of images to save in the sample output.
    sample_size = 25

    # Variables to accumulate the mean squared error across all batches and devices.
    mse_sum = 0
    mse_n = 0

    # Loop over the data loader.
    for i, (img, label) in enumerate(loader):
        # Reset the gradients for the new iteration.
        model.zero_grad()

        # Move the input image tensor to the specified device (e.g., GPU).
        img = img.to(device)

        # Forward pass: get the reconstructed image 'out' and the 'latent_loss'.
        out, latent_loss = model(img)
        # Calculate the reconstruction loss between the output and the original image.
        recon_loss = criterion(out, img)
        # The latent loss is averaged over the batch.
        latent_loss = latent_loss.mean()
        # Combine the reconstruction and latent losses to get the final loss.
        loss = recon_loss + latent_loss_weight * latent_loss
        # Backward pass: compute gradients.
        loss.backward()

        # If a learning rate scheduler is being used, take a step.
        if scheduler is not None:
            scheduler.step()
        # Update the model's parameters using the computed gradients.
        optimizer.step()

        # Get the reconstruction loss for the current batch.
        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        # Create a dictionary to hold the loss information for this process.
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        # In a distributed setup, gather the loss information from all processes.
        comm = dist.all_gather(comm)

        # Aggregate the loss information from all processes.
        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        # Only the primary process should handle logging.
        if dist.is_primary():
            # Get the current learning rate.
            lr = optimizer.param_groups[0]["lr"]

            # Set the description of the tqdm progress bar to show current training stats.
            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            # Periodically, save a sample of reconstructed images.
            if i % 100 == 0:
                # Set the model to evaluation mode. This disables layers like dropout.
                model.eval()

                # Take a small sample from the current batch.
                sample = img[:sample_size]

                # Perform inference without calculating gradients.
                with torch.no_grad():
                    out, _ = model(sample)

                # Save a grid of original and reconstructed images.
                utils.save_image(
                    torch.cat([sample, out], 0),
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    value_range=(-1, 1), # Normalize the image pixel values for saving.
                )

                # Set the model back to training mode.
                model.train()


def main(args):
    """
    ## Main Orchestration Function
    This function sets up and orchestrates the entire training process. It initializes
    the device, configures distributed training if applicable, prepares the dataset
    and data loaders, instantiates the VQ-VAE model, optimizer, and scheduler, and
    then calls the `train` function for the specified number of epochs.
    """
    # Set the device to 'cuda' for GPU training.
    device = "cuda"

    # Check if we are in a distributed environment (more than one process).
    args.distributed = dist.get_world_size() > 1

    # Define a pipeline of transformations to apply to the images.
    transform = transforms.Compose(
        [
            transforms.Resize(args.size), # Resize images to the specified size.
            transforms.CenterCrop(args.size), # Crop the center of the image.
            transforms.ToTensor(), # Convert images to PyTorch tensors.
            # Normalize tensors to a range of [-1, 1].
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # Load the dataset from the specified path using ImageFolder.
    dataset = datasets.ImageFolder(args.path, transform=transform)
    # Create a data sampler that handles shuffling and distributing data across GPUs.
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    # Create the DataLoader to feed data to the model in batches.
    loader = DataLoader(
        dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
    )

    # Instantiate the VQVAE model and move it to the specified device.
    model = VQVAE().to(device)

    # If in a distributed environment, wrap the model in DistributedDataParallel.
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()], # Assign model to a specific GPU.
            output_device=dist.get_local_rank(),
        )

    # Initialize the Adam optimizer with the model's parameters and learning rate.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Initialize scheduler to None.
    scheduler = None
    # If a scheduler is specified in the arguments, create it.
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    # The main training loop, which iterates over the number of epochs.
    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)

        # Only the primary process should save model checkpoints.
        if dist.is_primary():
            # Save the model's state dictionary.
            torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    """
    ## Script Execution Block
    This block is executed when the script is run directly. It is responsible for
    parsing command-line arguments that configure the training run, such as the
    dataset path, learning rate, and number of epochs. It then uses these arguments
    to launch the main training function, handling the setup for distributed training
    if multiple GPUs are specified.
    """
    # Create an argument parser to read command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    # A trick to get a free port for distributed training communication.
    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    # Define command-line arguments for training configuration.
    parser.add_argument("--size", type=int, default=256, help="image size to train on")
    parser.add_argument("--epoch", type=int, default=560, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--sched", type=str, help="learning rate scheduler (e.g., 'cycle')")
    parser.add_argument("path", type=str, help="path to the image dataset")

    # Parse the arguments provided from the command line.
    args = parser.parse_args()

    # Print the configuration arguments.
    print(args)

    # Launch the training process. This function handles spawning processes for distributed training.
    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))