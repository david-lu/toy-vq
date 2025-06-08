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
        # So everything in dim 0 is basically a patch
        flatten = input.reshape(-1, self.dim)

        #   B*N = number of vectors in the batch
        #   C = number of channels for each latent vector (e.g., 64)
        #   E = number of embeddings in the codebook (e.g., 512)

        # This entire block calculates the squared Euclidean distance: ||a - b||² = a² - 2ab + b²
        # where 'a' is 'flatten' and 'b' is 'self.embed'.
        dist = (
            # --- Term 1: a² ---
            # `flatten.pow(2)` -> shape [B*N, C]
            # `.sum(1, keepdim=True)` sums along the channel dimension C.
            # Resulting shape: [B*N, 1]
                flatten.pow(2).sum(1, keepdim=True)

                # --- Term 2: -2ab ---
                # This is the matrix multiplication of flatten and the codebook.
                # `flatten @ self.embed` -> [B*N, C] @ [C, E]
                # Resulting shape: [B*N, E]
                - 2 * flatten @ self.embed

                # --- Term 3: b² ---
                # `self.embed.pow(2)` -> shape [C, E]
                # `.sum(0, keepdim=True)` sums along the channel dimension C.
                # Resulting shape: [1, E]
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        # --- Final Output ---
        # Returned is the distance between each vector in the batch and each embedding in the codebook.
        # `dist` shape: [B*N, E]

        # Find the index of the closest embedding for each input vector.
        _, embed_ind = (-dist).max(1)
        # `embed_ind` shape: [B*N]
        #   - A 1D tensor containing the index of the closest codebook vector for each of the N input vectors.

        # Create a one-hot encoding of the embedding indices.
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        # `embed_onehot` shape: [B*N, E]
        #   - Each index is converted into a one-hot vector of length E.

        # Reshape the embedding indices to the spatial dimensions of the input.
        embed_ind = embed_ind.view(*input.shape[:-1])
        # `embed_ind` shape (reshaped): [B, H', W']
        #   - The flat list of N indices is reshaped into a spatially ordered map,
        #     preserving the batch and spatial structure.

        # Get the quantized vectors from the codebook using the indices.
        quantize = self.embed_code(embed_ind)
        # `quantize` shape: [B, H', W', C]
        #   - Each index in the [B, H', W'] map is replaced by its corresponding C-dimensional
        #     vector from the codebook, resulting in the final quantized latent map.

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
    This is a standard residual block (ResBlock). Its core idea is to learn a
    residual function F(x) and add its result back to the original input x.
    The final output is H(x) = F(x) + x.

    This structure, known as a "skip connection" or "identity shortcut," helps
    in training very deep networks by preventing the vanishing gradient problem
    and making it easier for layers to learn an identity mapping if needed.

    The convolutions inside this block are responsible for implementing F(x).
    They learn to refine the features from the input tensor.
    """
    def __init__(self, in_channel, channel):
        """
        Initializes the ResBlock.
        Args:
            in_channel (int): Number of input and output channels for the block.
            channel (int): Number of channels for the internal hidden layer.
        """
        super().__init__()

        # The `self.conv` sequence defines the residual function F(x).
        # It's a series of layers that transform the input to produce a "change" or "delta".
        # Convolutions are used because they are excellent at processing spatial data (like feature maps)
        # and learning hierarchies of patterns (edges, textures, etc.).
        self.conv = nn.Sequential(
            # First, apply a non-linear activation. This is a "pre-activation" design.
            nn.ReLU(),

            # The first convolution is the main feature-learning step.
            # It uses a 3x3 kernel to process a local neighborhood of the feature map.
            # It expands the number of channels from `in_channel` to `channel`, creating
            # a richer feature space for the block to work with.
            nn.Conv2d(in_channel, channel, 3, padding=1),

            # Apply another non-linear activation.
            nn.ReLU(inplace=True),

            # The second convolution is a 1x1 "projection" layer.
            # A 1x1 convolution acts like a per-pixel fully connected layer across channels.
            # Its crucial job here is to project the feature map from `channel` back down
            # to `in_channel`, ensuring its output shape matches the original input's shape
            # so they can be added together.
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        """
        Forward pass of the residual block.
        Args:
            input (Tensor): The input tensor, x.
        Returns:
            out (Tensor): The output tensor, H(x) = F(x) + x.
        """
        # Calculate the residual, F(x), by passing the input through the convolutional path.
        out = self.conv(input)

        # This is the residual connection (the "+ x" part).
        # The output of the convolutional layers (the residual) is added element-wise
        # to the original input.
        out += input

        return out


class Encoder(nn.Module):
    """
    ## Encoder Network
    The Encoder's job is to take a high-dimensional input (like an image) and
    compress it into a lower-dimensional spatial representation (a feature map).
    It achieves this by using a series of strided convolutions to downsample the
    input, while simultaneously increasing the number of channels to capture more
    complex features. The ResBlocks are used to build a deeper and more powerful
    network, allowing for more sophisticated feature extraction at each spatial level.
    """

    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        """
        Initializes the Encoder.
        Args:
            in_channel (int): Number of input channels (e.g., 3 for an RGB image).
            channel (int): Number of output channels for the main convolutions.
            n_res_block (int): Number of residual blocks to use.
            n_res_channel (int): Number of channels in the hidden layers of the ResBlocks.
            stride (int): The total downsampling factor (e.g., 4 or 2).
        """
        super().__init__()

        blocks = []

        if stride == 4:
            # This configuration downsamples the input by a total factor of 4.
            # Initial shape: [B, in_channel, H, W]
            blocks.extend([
                # First downsampling layer: Halves spatial dimensions.
                # Shape: [B, in_channel, H, W] -> [B, channel // 2, H/2, W/2]
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),

                # Second downsampling layer: Halves spatial dimensions again.
                # Shape: [B, channel // 2, H/2, W/2] -> [B, channel, H/4, W/4]
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),

                # A 3x3 convolution to refine features without changing dimensions.
                # Shape: [B, channel, H/4, W/4] -> [B, channel, H/4, W/4]
                nn.Conv2d(channel, channel, 3, padding=1),
            ])

        elif stride == 2:
            # This configuration downsamples the input by a factor of 2.
            # Initial shape: [B, in_channel, H, W]
            blocks.extend([
                # Single downsampling layer: Halves spatial dimensions.
                # Shape: [B, in_channel, H, W] -> [B, channel // 2, H/2, W/2]
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),

                # A 3x3 convolution to refine features and adjust channels.
                # Shape: [B, channel // 2, H/2, W/2] -> [B, channel, H/2, W/2]
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ])

        # Add a series of residual blocks.
        # These blocks do not change the shape of the tensor.
        # Shape remains: [B, channel, H_latent, W_latent] throughout the loop.
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        # A final activation function for the entire encoder block. Shape is unchanged.
        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        """
        Forward pass of the Encoder.
        Args:
            input (Tensor): Input tensor with shape [B, C_in, H, W].
        Returns:
            (Tensor): Downsampled latent feature map with shape [B, C, H/stride, W/stride].
        """
        return self.blocks(input)


class Decoder(nn.Module):
    """
    ## Decoder Network
    The Decoder's purpose is to perform the inverse operation of the Encoder.
    It takes a low-dimensional spatial feature map (the quantized latent representation)
    and upsamples it back to the original input's dimensions (e.g., an image).
    It uses Transposed Convolutions for upsampling and ResBlocks to refine the
    features at each spatial level, helping to generate a high-quality reconstruction.
    """

    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        """
        Initializes the Decoder.
        Args:
            in_channel (int): Number of channels in the input latent map.
            out_channel (int): Number of channels in the final output image (e.g., 3 for RGB).
            channel (int): Number of channels for the main convolutions.
            n_res_block (int): Number of residual blocks to use.
            n_res_channel (int): Number of channels in the hidden layers of the ResBlocks.
            stride (int): The total upsampling factor (e.g., 4 or 2).
        """
        super().__init__()

        # Initial shape: [B, in_channel, H_latent, W_latent]
        blocks = [
            # Initial 3x3 convolution to transform the input latent features.
            # Shape: [B, in_channel, H_latent, W_latent] -> [B, channel, H_latent, W_latent]
            nn.Conv2d(in_channel, channel, 3, padding=1)
        ]

        # Add a series of residual blocks to refine features before upsampling.
        # These blocks do not change the shape of the tensor.
        # Shape remains: [B, channel, H_latent, W_latent] throughout the loop.
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        # Apply an activation function before upsampling. Shape is unchanged.
        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            # This configuration upsamples the input by a total factor of 4.
            blocks.extend(
                [
                    # First upsampling layer: Doubles spatial dimensions.
                    # Shape: [B, channel, H, W] -> [B, channel // 2, H*2, W*2]
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),

                    # Second upsampling layer: Doubles dimensions again to reach the target size.
                    # Shape: [B, channel // 2, H*2, W*2] -> [B, out_channel, H*4, W*4]
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            # This configuration upsamples the input by a factor of 2.
            # Shape: [B, channel, H, W] -> [B, out_channel, H*2, W*2]
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        """
        Forward pass of the Decoder.
        Args:
            input (Tensor): Input tensor (latent map) with shape [B, C_in, H, W].
        Returns:
            (Tensor): Reconstructed output with shape [B, C_out, H*stride, W*stride].
        """
        return self.blocks(input)


class VQVAE(nn.Module):
    """
    ## Hierarchical VQ-VAE Model
    This is the main VQ-VAE model that brings all the other components together.
    It implements a hierarchical structure with two levels of quantization (top and bottom).
    This allows the model to capture both high-level, abstract information (in the
    smaller, top-level latent map) and fine-grained details (in the larger,
    bottom-level latent map). This separation of concerns helps in generating
    high-fidelity images.
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
            in_channel (int): C_in, number of channels in the input image.
            channel (int): C, number of channels in the main convolutional layers.
            n_res_block (int): Number of residual blocks.
            n_res_channel (int): Number of channels in the residual blocks' hidden layers.
            embed_dim (int): C_emb, the dimension of the embedding vectors.
            n_embed (int): The number of embedding vectors in the codebook.
            decay (float): The decay factor for EMA updates in the Quantize layer.
        """
        super().__init__()

        # Bottom-level encoder, which downsamples by a factor of 4.
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        # Top-level encoder, which downsamples the bottom-level features by another 2x.
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        # 1x1 Conv to map top-level features to the embedding dimension.
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        # Top-level quantization layer.
        self.quantize_t = Quantize(embed_dim, n_embed)
        # Decoder for the top-level features, upsampling by 2x.
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        # 1x1 Conv to map the combined bottom and top features to the embedding dimension.
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        # Bottom-level quantization layer.
        self.quantize_b = Quantize(embed_dim, n_embed)
        # Upsampling layer for the top-level features to match bottom-level dimensions.
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        # The final decoder that reconstructs the image from the combined latent maps.
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
            input (Tensor): The input image tensor with shape [B, C_in, H, W].
        Returns:
            dec (Tensor): The reconstructed image with shape [B, C_in, H, W].
            diff (Tensor): The total commitment loss (scalar).
        """
        # Encode the input to get the quantized representations and commitment loss.
        quant_t, quant_b, diff, _, _ = self.encode(input)
        # Decode the quantized representations to reconstruct the image.
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        """
        Encodes the input image into discrete latent codes for both hierarchy levels.
        Args:
            input (Tensor): The input image tensor with shape [B, C_in, H, W].
        Returns:
            quant_t (Tensor): Quantized top-level latent map, shape [B, C_emb, H/8, W/8].
            quant_b (Tensor): Quantized bottom-level latent map, shape [B, C_emb, H/4, W/4].
            diff (Tensor): Total commitment loss.
            id_t (Tensor): Top-level indices, shape [B, H/8, W/8].
            id_b (Tensor): Bottom-level indices, shape [B, H/4, W/4].
        """
        # --- Top-Level Encoding ---
        # Initial shape: [B, C_in, H, W]
        enc_b = self.enc_b(input) # Shape: [B, C, H/4, W/4]
        enc_t = self.enc_t(enc_b) # Shape: [B, C, H/8, W/8]

        # --- Top-Level Quantization ---
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1) # Shape: [B, H/8, W/8, C_emb]
        quant_t, diff_t, id_t = self.quantize_t(quant_t) # quant_t shape: [B, H/8, W/8, C_emb]
        quant_t = quant_t.permute(0, 3, 1, 2) # Shape: [B, C_emb, H/8, W/8]
        diff_t = diff_t.unsqueeze(0)

        # --- Bottom-Level Encoding ---
        # Decode the top-level quantized map to provide context for the bottom level.
        dec_t = self.dec_t(quant_t) # Shape: [B, C_emb, H/4, W/4]
        # Concatenate the decoded top features with the original bottom features along the channel axis.
        enc_b = torch.cat([dec_t, enc_b], 1) # Shape: [B, C_emb + C, H/4, W/4]

        # --- Bottom-Level Quantization ---
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1) # Shape: [B, H/4, W/4, C_emb]
        quant_b, diff_b, id_b = self.quantize_b(quant_b) # quant_b shape: [B, H/4, W/4, C_emb]
        quant_b = quant_b.permute(0, 3, 1, 2) # Shape: [B, C_emb, H/4, W/4]
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        """
        Decodes the quantized latent representations from both levels into an image.
        Args:
            quant_t (Tensor): Quantized top-level latent map, shape [B, C_emb, H/8, W/8].
            quant_b (Tensor): Quantized bottom-level latent map, shape [B, C_emb, H/4, W/4].
        Returns:
            dec (Tensor): The reconstructed image, shape [B, C_in, H, W].
        """
        # Upsample the top-level latent map to match the bottom-level's spatial dimensions.
        upsample_t = self.upsample_t(quant_t) # Shape: [B, C_emb, H/4, W/4]
        # Concatenate the upsampled top and the bottom latent maps along the channel axis.
        quant = torch.cat([upsample_t, quant_b], 1) # Shape: [B, C_emb + C_emb, H/4, W/4]
        # Pass the combined latent map through the final decoder to reconstruct the image.
        dec = self.dec(quant) # Shape: [B, C_in, H, W]

        return dec

    def decode_code(self, code_t, code_b):
        """
        Decodes from discrete latent codes (indices) to an image. Used for generation.
        Args:
            code_t (Tensor): Top-level indices, shape [B, H/8, W/8].
            code_b (Tensor): Bottom-level indices, shape [B, H/4, W/4].
        Returns:
            dec (Tensor): The generated image, shape [B, C_in, H, W].
        """
        # Look up the embedding vectors for the top-level codes.
        quant_t = self.quantize_t.embed_code(code_t) # Shape: [B, H/8, W/8, C_emb]
        quant_t = quant_t.permute(0, 3, 1, 2) # Shape: [B, C_emb, H/8, W/8]
        # Look up the embedding vectors for the bottom-level codes.
        quant_b = self.quantize_b.embed_code(code_b) # Shape: [B, H/4, W/4, C_emb]
        quant_b = quant_b.permute(0, 3, 1, 2) # Shape: [B, C_emb, H/4, W/4]

        # Use the standard decode method to generate the image from the looked-up vectors.
        dec = self.decode(quant_t, quant_b)

        return dec