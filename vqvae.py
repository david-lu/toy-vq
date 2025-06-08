import torch
from torch import nn
from torch.nn import functional as F

# It's assumed that 'distributed' is a helper module for distributed training.
# The functions within would handle communication between different processes/machines.
import distributed as dist_fn


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    """
    ## Vector Quantization Layer
    This block is the core of the "Vector Quantized" (VQ) part of the VQ-VAE.
    It takes the continuous output of the encoder and maps each vector to the
    closest vector in a learned, finite-sized codebook (embedding space).
    This process discretizes the latent representation. It also implements the
    straight-through estimator to allow gradients to flow back to the encoder
    during training and calculates the commitment loss to train the codebook.
    """
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        """
        Initializes the Quantize layer.
        Args:
            dim (int): The dimension of the input vectors and the embedding vectors.
            n_embed (int): The number of embedding vectors in the codebook.
            decay (float): The decay factor for the exponential moving average update of the embeddings.
            eps (float): A small value to prevent division by zero during normalization.
        """
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        # 'embed' is the codebook of the VQ-VAE. It's a learnable tensor.
        # It's of size (dim, n_embed)
        embed = torch.randn(dim, n_embed)
        # 'register_buffer' is used to store tensors that are part of the model's state,
        # but are not parameters to be trained by the optimizer.
        self.register_buffer("embed", embed)
        # 'cluster_size' keeps track of the size of each cluster in the codebook using an exponential moving average.
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        # 'embed_avg' is the running average of the vectors in each cluster.
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        """
        Forward pass of the quantization layer.
        Args:
            input (Tensor): The input tensor from the encoder, of shape (B, H, W, C) or (B, N, C).
        Returns:
            quantize (Tensor): The quantized output tensor, with the same shape as the input.
            diff (Tensor): The commitment loss.
            embed_ind (Tensor): The indices of the closest embeddings in the codebook.
        """
        # Reshape the input to be a 2D tensor of shape (B*H*W, C) or (B*N, C)
        flatten = input.reshape(-1, self.dim)
        # Calculate the L2 distance between each input vector and each embedding vector.
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        # Find the index of the closest embedding for each input vector.
        _, embed_ind = (-dist).max(1)
        # Create a one-hot encoding of the embedding indices.
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        # Reshape the embedding indices to the spatial dimensions of the input.
        embed_ind = embed_ind.view(*input.shape[:-1])
        # Get the quantized vectors from the codebook using the indices.
        quantize = self.embed_code(embed_ind)

        # During training, update the codebook embeddings.
        if self.training:
            # Sum of one-hot encoded vectors for each embedding.
            embed_onehot_sum = embed_onehot.sum(0)
            # Sum of input vectors that are closest to each embedding.
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # In a distributed setting, sum the statistics across all devices.
            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            # Update the cluster sizes using an exponential moving average.
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            # Update the average embedding vectors using an exponential moving average.
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            # Normalize the cluster sizes.
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            # Get the updated embedding vectors by dividing the sum of vectors by the cluster size.
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            # Copy the updated embeddings to the codebook.
            self.embed.data.copy_(embed_normalized)

        # The commitment loss, encouraging the encoder output to be close to the chosen codebook vector.
        # .detach() is used to stop gradients from flowing back into the encoder from this loss.
        diff = (quantize.detach() - input).pow(2).mean()
        # The straight-through estimator: gradients from 'quantize' are copied to 'input'.
        # This allows gradients to flow back through the quantization layer to the encoder.
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        """
        Retrieves the embedding vectors from the codebook given their indices.
        Args:
            embed_id (Tensor): A tensor of indices.
        Returns:
            (Tensor): The corresponding embedding vectors.
        """
        # F.embedding is a lookup function.
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    """
    ## Residual Block
    This is a standard residual block (ResBlock). Its purpose is to help build
    deeper and more powerful encoder and decoder networks. By adding its input
    to its output (a "skip connection"), it helps prevent the vanishing gradient
    problem, making it easier to train the overall model.
    """
    def __init__(self, in_channel, channel):
        """
        Initializes the ResBlock.
        Args:
            in_channel (int): Number of input channels.
            channel (int): Number of channels in the hidden convolutional layer.
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        """
        Forward pass of the residual block.
        Args:
            input (Tensor): The input tensor.
        Returns:
            out (Tensor): The output tensor.
        """
        out = self.conv(input)
        # The output of the convolutional layers is added to the original input.
        out += input

        return out


class Encoder(nn.Module):
    """
    ## Encoder Network
    The Encoder's job is to take an input, like an image, and compress it.
    It uses a series of convolutional and residual blocks to downsample the
    input into a smaller, lower-resolution feature map. This compressed
    feature map is the continuous latent representation that will be fed into
    the Quantize layer.
    """
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        """
        Initializes the Encoder.
        Args:
            in_channel (int): Number of input channels (e.g., 3 for an RGB image).
            channel (int): Number of channels in the main convolutional layers.
            n_res_block (int): Number of residual blocks.
            n_res_channel (int): Number of channels in the residual blocks' hidden layers.
            stride (int): The stride of the convolutional layers, controlling the downsampling factor.
        """
        super().__init__()

        # These blocks are for downsampling the input.
        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        # Add the specified number of residual blocks.
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        """
        Forward pass of the Encoder.
        Args:
            input (Tensor): The input tensor.
        Returns:
            (Tensor): The encoded latent representation.
        """
        return self.blocks(input)


class Decoder(nn.Module):
    """
    ## Decoder Network
    The Decoder does the opposite of the Encoder. It takes the discretized,
    quantized latent representation and upsamples it back to the original
    input's dimensions. It uses a series of transposed convolutional layers
    and residual blocks to reconstruct the data (e.g., an image) from the
    compressed representation.
    """
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        """
        Initializes the Decoder.
        Args:
            in_channel (int): Number of input channels for the decoder (dimension of the latent space).
            out_channel (int): Number of output channels (e.g., 3 for an RGB image).
            channel (int): Number of channels in the main convolutional layers.
            n_res_block (int): Number of residual blocks.
            n_res_channel (int): Number of channels in the residual blocks' hidden layers.
            stride (int): The stride of the transposed convolutional layers, controlling the upsampling factor.
        """
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        # Add the specified number of residual blocks.
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        # These blocks are for upsampling the latent representation.
        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        """
        Forward pass of the Decoder.
        Args:
            input (Tensor): The latent representation.
        Returns:
            (Tensor): The reconstructed output.
        """
        return self.blocks(input)


class VQVAE(nn.Module):
    """
    ## Hierarchical VQ-VAE Model
    This is the main VQ-VAE model that brings all the other components together.
    It implements a hierarchical structure with two levels (top and bottom) of
    encoding and quantization. This allows the model to capture both high-level,
    abstract information (top level) and fine-grained details (bottom level).
    The class defines the complete forward pass, from encoding the input image
    to decoding the quantized codes back into a reconstructed image.
    """
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        """
        Initializes the VQ-VAE model.
        Args:
            in_channel (int): Number of channels in the input image.
            channel (int): Number of channels in the main convolutional layers.
            n_res_block (int): Number of residual blocks.
            n_res_channel (int): Number of channels in the residual blocks' hidden layers.
            embed_dim (int): The dimension of the embedding vectors.
            n_embed (int): The number of embedding vectors in the codebook.
            decay (float): The decay factor for the exponential moving average update of the embeddings.
        """
        super().__init__()

        # Bottom-level encoder, which downsamples by a factor of 4.
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        # Top-level encoder, which downsamples the output of the bottom encoder by a factor of 2.
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        # Convolutional layer to prepare the top-level latent representation for quantization.
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        # Top-level quantization layer.
        self.quantize_t = Quantize(embed_dim, n_embed)
        # Top-level decoder to reconstruct the top-level latent representation.
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        # Convolutional layer to prepare the bottom-level latent representation for quantization.
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        # Bottom-level quantization layer.
        self.quantize_b = Quantize(embed_dim, n_embed)
        # Transposed convolutional layer to upsample the top-level quantized output.
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        # Final decoder to reconstruct the image from the combined quantized representations.
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        """
        The main forward pass for the VQ-VAE, used during training.
        Args:
            input (Tensor): The input image tensor.
        Returns:
            dec (Tensor): The reconstructed image.
            diff (Tensor): The total commitment loss (sum of top and bottom).
        """
        # Encode the input to get the quantized representations and commitment loss.
        quant_t, quant_b, diff, _, _ = self.encode(input)
        # Decode the quantized representations to reconstruct the image.
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        """
        Encodes the input image into discrete latent codes.
        Args:
            input (Tensor): The input image tensor.
        Returns:
            quant_t (Tensor): The quantized top-level representation.
            quant_b (Tensor): The quantized bottom-level representation.
            diff_t + diff_b (Tensor): The total commitment loss.
            id_t (Tensor): The top-level discrete codes.
            id_b (Tensor): The bottom-level discrete codes.
        """
        # Bottom-level encoding.
        enc_b = self.enc_b(input)
        # Top-level encoding.
        enc_t = self.enc_t(enc_b)

        # Prepare for top-level quantization.
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        # Top-level quantization.
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        # Permute back to (B, C, H, W).
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        # Decode the top-level quantized representation.
        dec_t = self.dec_t(quant_t)
        # Concatenate the decoded top-level representation with the bottom-level encoding.
        enc_b = torch.cat([dec_t, enc_b], 1)

        # Prepare for bottom-level quantization.
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        # Bottom-level quantization.
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        # Permute back to (B, C, H, W).
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        """
        Decodes the quantized representations to reconstruct the image.
        Args:
            quant_t (Tensor): The quantized top-level representation.
            quant_b (Tensor): The quantized bottom-level representation.
        Returns:
            dec (Tensor): The reconstructed image.
        """
        # Upsample the top-level representation.
        upsample_t = self.upsample_t(quant_t)
        # Concatenate the upsampled top-level and the bottom-level representations.
        quant = torch.cat([upsample_t, quant_b], 1)
        # Decode the combined representation.
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        """
        Decodes from discrete latent codes to an image.
        This is useful for generating new images from learned codes.
        Args:
            code_t (Tensor): The top-level discrete codes.
            code_b (Tensor): The bottom-level discrete codes.
        Returns:
            dec (Tensor): The generated image.
        """
        # Get the embedding vectors for the top-level codes.
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        # Get the embedding vectors for the bottom-level codes.
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        # Decode the embedding vectors to an image.
        dec = self.decode(quant_t, quant_b)

        return dec