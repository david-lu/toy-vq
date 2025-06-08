# Import necessary libraries
import argparse  # For parsing command-line arguments

import numpy as np  # For numerical operations
import torch  # The main deep learning framework
from torch import nn, optim  # Neural network modules and optimization algorithms
from torch.utils.data import DataLoader  # For creating data loaders
from tqdm import tqdm  # For creating progress bars

# Try to import Apex for automatic mixed-precision training, which can speed up
# training and reduce memory usage on NVIDIA GPUs. If not available, set amp to None.
try:
    from apex import amp

except ImportError:
    amp = None

# Import custom modules from the project
from dataset import LMDBDataset  # A custom dataset class for reading from LMDB files
from pixelsnail import PixelSNAIL  # The PixelSNAIL model implementation
from scheduler import CycleScheduler  # A custom learning rate scheduler


# --- Training Function for a Single Epoch ---
def train(args, epoch, loader, model, optimizer, scheduler, device):
    """
    This function handles the training loop for one epoch.

    Args:
        args: Command-line arguments.
        epoch (int): The current epoch number.
        loader (DataLoader): The data loader for the training set.
        model (nn.Module): The PixelSNAIL model.
        optimizer (optim.Optimizer): The optimizer.
        scheduler: The learning rate scheduler.
        device (str): The device to train on ('cuda' or 'cpu').
    """
    # Wrap the data loader with tqdm to create a command-line progress bar
    loader = tqdm(loader)

    # CrossEntropyLoss is used because we are predicting the next discrete code (a classification task)
    criterion = nn.CrossEntropyLoss()

    # Iterate over the data loader, which yields batches of hierarchical codes
    for i, (top, bottom, label) in enumerate(loader):
        # Reset gradients from the previous iteration
        model.zero_grad()

        # Move the top-level codes to the specified device
        top = top.to(device)

        # --- Hierarchical Logic ---
        # This block determines whether to train the top-level prior or the bottom-level conditional prior.
        if args.hier == 'top':
            # Target is the top-level codes themselves
            target = top
            # Model predicts the next top-level code based on previous ones
            out, _ = model(top)

        elif args.hier == 'bottom':
            # Move bottom-level codes to the device
            bottom = bottom.to(device)
            # Target is the bottom-level codes
            target = bottom
            # Model predicts the next bottom-level code conditioned on the top-level codes
            out, _ = model(bottom, condition=top)

        # Calculate the loss between the model's predictions and the true next code
        loss = criterion(out, target)
        # Perform backpropagation to compute gradients
        loss.backward()

        # If a scheduler is used, update the learning rate
        if scheduler is not None:
            scheduler.step()
        # Update the model's weights
        optimizer.step()

        # --- Metrics Calculation ---
        # Get the predicted code index by finding the max logit
        _, pred = out.max(1)
        # Calculate the accuracy by comparing predictions to the target
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        # Get the current learning rate from the optimizer
        lr = optimizer.param_groups[0]['lr']

        # Update the progress bar description with the current loss, accuracy, and learning rate
        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )


# A simple transform class to convert NumPy arrays to PyTorch LongTensors.
# Note: This is defined but not explicitly used in the main block, suggesting the
# LMDBDataset class handles this transformation internally.
class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        # Convert the input (likely an image or data array) to a NumPy array
        ar = np.array(input)

        # Convert the NumPy array to a PyTorch tensor of type Long
        return torch.from_numpy(ar).long()


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Argument Parsing ---
    # Sets up the argument parser to read configurations from the command line.
    parser = argparse.ArgumentParser()
    # Data and training parameters
    parser.add_argument('--batch', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epoch', type=int, default=420, help='Number of epochs to train')
    parser.add_argument('--hier', type=str, default='top', help='Which hierarchy to train: "top" or "bottom"')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    # Model architecture parameters
    parser.add_argument('--channel', type=int, default=256, help='Number of channels in the model')
    parser.add_argument('--n_res_block', type=int, default=4, help='Number of residual blocks in each PixelBlock')
    parser.add_argument('--n_res_channel', type=int, default=256, help='Number of channels in the residual blocks')
    parser.add_argument('--n_out_res_block', type=int, default=0, help='Number of residual blocks in the output head')
    parser.add_argument('--n_cond_res_block', type=int, default=3,
                        help='Number of residual blocks in the conditioning network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    # Technical parameters
    parser.add_argument('--amp', type=str, default='O0', help='Apex AMP optimization level')
    parser.add_argument('--sched', type=str, help='Name of the learning rate scheduler to use (e.g., "cycle")')
    parser.add_argument('--ckpt', type=str, help='Path to a checkpoint file to resume training')
    parser.add_argument('path', type=str, help='Path to the LMDB dataset')

    args = parser.parse_args()

    print(args)

    device = 'cuda'  # Set the device to CUDA for GPU training

    # --- Data Loading ---
    # Initialize the custom LMDB dataset
    dataset = LMDBDataset(args.path)
    # Create a DataLoader for batching, shuffling, and multi-threaded data loading
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )

    ckpt = {}

    # --- Checkpoint Loading ---
    # If a checkpoint path is provided, load it to resume training.
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        # Restore the arguments from the checkpoint to ensure model consistency
        args = ckpt['args']

    # --- Model Initialization ---
    # This logic builds the correct PixelSNAIL model based on the hierarchy.
    if args.hier == 'top':
        # The top-level model works on a 32x32 latent grid (from VQ-VAE)
        # with 512 possible code values.
        model = PixelSNAIL(
            [32, 32],  # Shape of the latent map
            512,  # Number of possible values for each code (n_embed)
            args.channel,
            5,  # Kernel size
            4,  # Number of PixelSNAIL blocks
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif args.hier == 'bottom':
        # The bottom-level model works on a 64x64 latent grid and is conditioned
        # on the top-level codes. Attention is turned off, likely for efficiency.
        model = PixelSNAIL(
            [64, 64],  # Shape of the latent map
            512,  # Number of possible values for each code
            args.channel,
            5,  # Kernel size
            4,  # Number of PixelSNAIL blocks
            args.n_res_block,
            args.n_res_channel,
            attention=False,  # Attention is disabled for the larger bottom level
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,  # Enable conditioning network
            cond_res_channel=args.n_res_channel,
        )

    # If the checkpoint contains model weights, load them.
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])

    model = model.to(device)  # Move the model to the GPU
    # Initialize the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Mixed Precision and Data Parallelism ---
    # If Apex is available, initialize it for mixed-precision training.
    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    # Wrap the model in DataParallel to use multiple GPUs on a single machine.
    model = nn.DataParallel(model)
    model = model.to(device)

    # --- Scheduler and Training Loop ---
    scheduler = None
    # If 'cycle' scheduler is specified, initialize it.
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    # The main loop that runs training for the specified number of epochs.
    for i in range(args.epoch):
        train(args, i, loader, model, optimizer, scheduler, device)
        # Save a checkpoint after each epoch.
        # 'model.module.state_dict()' is used to access the model's weights
        # from inside the DataParallel wrapper.
        torch.save(
            {'model': model.module.state_dict(), 'args': args},
            f'checkpoint/pixelsnail_{args.hier}_{str(i + 1).zfill(3)}.pt',
        )