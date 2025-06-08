# Copyright (c) Xi Chen
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Borrowed from https://github.com/neocxi/pixelsnail-public and ported it to PyTorch

from math import sqrt
from functools import partial, lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def wn_linear(in_dim, out_dim):
    """
    ## Weight-Normalized Linear Layer
    A helper function that creates a linear layer with weight normalization applied.
    Weight normalization can help stabilize and accelerate the training of neural networks.
    """
    return nn.utils.weight_norm(nn.Linear(in_dim, out_dim))


class WNConv2d(nn.Module):
    """
    ## Weight-Normalized 2D Convolutional Layer
    A wrapper class for a 2D convolutional layer that applies weight normalization.
    This is used throughout the model as a basic building block instead of a
    standard Conv2d layer to improve training dynamics.
    """
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        activation=None,
    ):
        super().__init__()

        # The core convolutional layer is wrapped with weight normalization.
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )

        # Store output channels and kernel size.
        self.out_channel = out_channel
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        self.kernel_size = kernel_size

        # An optional activation function can be applied after the convolution.
        self.activation = activation

    def forward(self, input):
        # Apply the convolution.
        out = self.conv(input)

        # Apply the activation if it exists.
        if self.activation is not None:
            out = self.activation(out)

        return out


def shift_down(input, size=1):
    """
    ## Shift Down Operation
    Helper function to shift the input tensor down by padding the top.
    This is used to create the vertical stack in the PixelSNAIL architecture.
    """
    return F.pad(input, [0, 0, size, 0])[:, :, : input.shape[2], :]


def shift_right(input, size=1):
    """
    ## Shift Right Operation
    Helper function to shift the input tensor to the right by padding the left.
    This is used to create the horizontal stack in the PixelSNAIL architecture.
    """
    return F.pad(input, [size, 0, 0, 0])[:, :, :, : input.shape[3]]


class CausalConv2d(nn.Module):
    """
    ## Causal 2D Convolution
    This module implements a causal (or masked) 2D convolution. To maintain the
    autoregressive property, the prediction for a given pixel can only depend on
    previously generated pixels (i.e., pixels above it and to its left in the
    raster scan order). This is achieved by padding the input and, in some cases,
    explicitly zeroing out parts of the convolutional kernel.
    """
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding='downright',
        activation=None,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2
        self.kernel_size = kernel_size

        # Determine the padding based on the causal dependency direction.
        if padding == 'downright':
            # Standard causal conv for PixelCNN.
            pad = [kernel_size[1] - 1, 0, kernel_size[0] - 1, 0]
        elif padding == 'down' or padding == 'causal':
            # Causal conv for the vertical or horizontal stack.
            pad = kernel_size[1] // 2
            pad = [pad, pad, kernel_size[0] - 1, 0]

        # In 'causal' mode, we need to manually mask the convolution kernel.
        self.causal = 0
        if padding == 'causal':
            self.causal = kernel_size[1] // 2

        # Create the padding layer.
        self.pad = nn.ZeroPad2d(pad)

        # The underlying convolution uses the WNConv2d module.
        self.conv = WNConv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=0, # Padding is handled by self.pad.
            activation=activation,
        )

    def forward(self, input):
        out = self.pad(input)

        # If 'causal' mode is enabled, manually zero out the weights that would
        # look at "future" pixels within the receptive field.
        if self.causal > 0:
            self.conv.conv.weight_v.data[:, :, -1, self.causal :].zero_()

        out = self.conv(out)
        return out


class GatedResBlock(nn.Module):
    """
    ## Gated Residual Block
    This block is a key component of the model, implementing a residual connection
    with a gating mechanism. The gate (a GLU layer) controls the flow of information
    through the block. It can also incorporate auxiliary inputs and conditioning
    information, making it a very flexible and powerful building block.
    """
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        conv='wnconv2d',
        activation=nn.ELU,
        dropout=0.1,
        auxiliary_channel=0,
        condition_dim=0,
    ):
        super().__init__()

        # Use partial to pre-configure the convolution type with specific padding.
        if conv == 'wnconv2d':
            conv_module = partial(WNConv2d, padding=kernel_size // 2)
        elif conv == 'causal_downright':
            conv_module = partial(CausalConv2d, padding='downright')
        elif conv == 'causal':
            conv_module = partial(CausalConv2d, padding='causal')

        self.activation = activation()
        # First convolutional layer.
        self.conv1 = conv_module(in_channel, channel, kernel_size)

        # Optional convolutional layer for an auxiliary input.
        if auxiliary_channel > 0:
            self.aux_conv = WNConv2d(auxiliary_channel, channel, 1)

        self.dropout = nn.Dropout(dropout)

        # Second convolutional layer. The output channels are doubled for the GLU gate.
        self.conv2 = conv_module(channel, in_channel * 2, kernel_size)

        # Optional 1x1 convolution for conditioning information.
        if condition_dim > 0:
            self.condition = WNConv2d(condition_dim, in_channel * 2, 1, bias=False)

        # Gated Linear Unit, which acts as a gate on the channel dimension.
        self.gate = nn.GLU(1)

    def forward(self, input, aux_input=None, condition=None):
        # Pass through the first convolution.
        out = self.conv1(self.activation(input))

        # Add auxiliary information if provided.
        if aux_input is not None:
            out = out + self.aux_conv(self.activation(aux_input))

        # Apply activation and dropout.
        out = self.activation(out)
        out = self.dropout(out)
        # Pass through the second convolution.
        out = self.conv2(out)

        # Add conditioning information if provided.
        if condition is not None:
            condition = self.condition(condition)
            out += condition

        # Apply the gating mechanism.
        out = self.gate(out)
        # Apply the residual connection.
        out += input

        return out


@lru_cache(maxsize=64)
def causal_mask(size):
    """
    ## Causal Mask Generator
    This function generates and caches a causal mask for self-attention. The mask
    ensures that a position can only attend to previous positions in the raster
    scan order, not future ones. The `@lru_cache` decorator memoizes the function,
    preventing re-computation of the same mask, which is a key optimization.
    """
    shape = [size, size]
    # Create an upper triangular matrix, which masks future positions.
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T
    # Create a start mask to prevent the first pixel from attending to anything.
    start_mask = np.ones(size).astype(np.float32)
    start_mask[0] = 0

    return (
        torch.from_numpy(mask).unsqueeze(0),
        torch.from_numpy(start_mask).unsqueeze(1),
    )


class CausalAttention(nn.Module):
    """
    ## Causal Self-Attention Module
    This module implements multi-head causal self-attention. Unlike convolutions,
    which have a limited receptive field, attention can model long-range dependencies
    between pixels. The causal mask ensures that the autoregressive property is
    maintained. This is the "attentive" part of the PixelSNAIL model.
    """
    def __init__(self, query_channel, key_channel, channel, n_head=8, dropout=0.1):
        super().__init__()

        # Linear layers to project inputs into query, key, and value spaces.
        self.query = wn_linear(query_channel, channel)
        self.key = wn_linear(key_channel, channel)
        self.value = wn_linear(key_channel, channel)

        self.dim_head = channel // n_head
        self.n_head = n_head

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key):
        batch, _, height, width = key.shape

        # Helper function to reshape tensors for multi-head attention.
        def reshape(input):
            return input.view(batch, -1, self.n_head, self.dim_head).transpose(1, 2)

        # Flatten the spatial dimensions of the query and key.
        query_flat = query.view(batch, query.shape[1], -1).transpose(1, 2)
        key_flat = key.view(batch, key.shape[1], -1).transpose(1, 2)

        # Project and reshape query, key, and value.
        query = reshape(self.query(query_flat))
        key = reshape(self.key(key_flat)).transpose(2, 3) # Transpose for matmul.
        value = reshape(self.value(key_flat))

        # Scaled dot-product attention.
        attn = torch.matmul(query, key) / sqrt(self.dim_head)
        # Get the causal mask for the flattened spatial dimension.
        mask, start_mask = causal_mask(height * width)
        mask = mask.type_as(query)
        start_mask = start_mask.type_as(query)
        # Apply the mask by setting future positions to a large negative number.
        attn = attn.masked_fill(mask == 0, -1e4)
        # Apply softmax and the start mask.
        attn = torch.softmax(attn, 3) * start_mask
        attn = self.dropout(attn)

        # Compute the weighted sum of values.
        out = attn @ value
        # Reshape the output back to the original spatial dimensions.
        out = out.transpose(1, 2).reshape(
            batch, height, width, self.dim_head * self.n_head
        )
        out = out.permute(0, 3, 1, 2)

        return out


class PixelBlock(nn.Module):
    """
    ## PixelSNAIL Block
    This module is a primary building block of the PixelSNAIL model. It consists
    of a series of GatedResBlocks to capture local dependencies, followed by an
    optional causal self-attention mechanism to capture global, long-range
    dependencies. This combination of convolutions and attention is what makes the
    "SNAIL" (Simple Neural Attentive Image Layer) architecture effective.
    """
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        n_res_block,
        attention=True,
        dropout=0.1,
        condition_dim=0,
    ):
        super().__init__()

        # A stack of gated residual blocks.
        resblocks = []
        for i in range(n_res_block):
            resblocks.append(
                GatedResBlock(
                    in_channel,
                    channel,
                    kernel_size,
                    conv='causal',
                    dropout=dropout,
                    condition_dim=condition_dim,
                )
            )
        self.resblocks = nn.ModuleList(resblocks)

        self.attention = attention

        if attention:
            # Residual blocks to process inputs for the key and query.
            self.key_resblock = GatedResBlock(
                in_channel * 2 + 2, in_channel, 1, dropout=dropout
            )
            self.query_resblock = GatedResBlock(
                in_channel + 2, in_channel, 1, dropout=dropout
            )

            # The causal attention module.
            self.causal_attention = CausalAttention(
                in_channel + 2, in_channel * 2 + 2, in_channel // 2, dropout=dropout
            )

            # An output residual block to merge the attention output.
            self.out_resblock = GatedResBlock(
                in_channel,
                in_channel,
                1,
                auxiliary_channel=in_channel // 2, # Attention output is the auxiliary input.
                dropout=dropout,
            )
        else:
            # If not using attention, just use a simple convolution.
            self.out = WNConv2d(in_channel + 2, in_channel, 1)

    def forward(self, input, background, condition=None):
        out = input

        # Pass through the stack of residual blocks.
        for resblock in self.resblocks:
            out = resblock(out, condition=condition)

        # If using attention...
        if self.attention:
            # Prepare key and query by concatenating inputs and positional encodings.
            key_cat = torch.cat([input, out, background], 1)
            key = self.key_resblock(key_cat)
            query_cat = torch.cat([out, background], 1)
            query = self.query_resblock(query_cat)
            # Perform causal attention.
            attn_out = self.causal_attention(query, key)
            # Merge the attention output with the convolutional output.
            out = self.out_resblock(out, attn_out)
        else:
            # If not using attention, just concatenate with background and convolve.
            bg_cat = torch.cat([out, background], 1)
            out = self.out(bg_cat)

        return out


class CondResNet(nn.Module):
    """
    ## Conditional Residual Network
    This is a small residual network used to process conditional information (e.g.,
    class labels or a low-resolution image) into a feature map. This feature map
    is then injected into the main PixelSNAIL blocks to guide the generation process.
    """
    def __init__(self, in_channel, channel, kernel_size, n_res_block):
        super().__init__()

        # A simple stack of a convolution and several gated residual blocks.
        blocks = [WNConv2d(in_channel, channel, kernel_size, padding=kernel_size // 2)]
        for i in range(n_res_block):
            blocks.append(GatedResBlock(channel, channel, kernel_size))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class PixelSNAIL(nn.Module):
    """
    ## PixelSNAIL Model
    This is the complete PixelSNAIL model. It combines all the building blocks to
    form a powerful autoregressive model for discrete data, such as the latent
    codes from a VQ-VAE. It works by factorizing the joint distribution of pixels
    into a product of conditional distributions, modeled by the neural network.
    The architecture uses two initial causal convolution streams (vertical and
    horizontal) followed by a series of PixelBlocks that use both convolutions
s    and self-attention to model dependencies.
    """
    def __init__(
        self,
        shape,
        n_class,
        channel,
        kernel_size,
        n_block,
        n_res_block,
        res_channel,
        attention=True,
        dropout=0.1,
        n_cond_res_block=0,
        cond_res_channel=0,
        cond_res_kernel=3,
        n_out_res_block=0,
    ):
        super().__init__()

        height, width = shape
        self.n_class = n_class

        # Ensure the kernel size for the initial convolutions is odd.
        if kernel_size % 2 == 0:
            kernel = kernel_size + 1
        else:
            kernel = kernel_size

        # Initial causal convolutions for the horizontal and vertical stacks.
        self.horizontal = CausalConv2d(
            n_class, channel, [kernel // 2, kernel], padding='down'
        )
        self.vertical = CausalConv2d(
            n_class, channel, [(kernel + 1) // 2, kernel // 2], padding='downright'
        )

        # Create and register fixed positional encodings.
        coord_x = (torch.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer('background', torch.cat([coord_x, coord_y], 1))

        # Main stack of PixelBlocks.
        self.blocks = nn.ModuleList()
        for i in range(n_block):
            self.blocks.append(
                PixelBlock(
                    channel,
                    res_channel,
                    kernel_size,
                    n_res_block,
                    attention=attention,
                    dropout=dropout,
                    condition_dim=cond_res_channel,
                )
            )

        # Optional ResNet for processing conditional inputs.
        if n_cond_res_block > 0:
            self.cond_resnet = CondResNet(
                n_class, cond_res_channel, cond_res_kernel, n_cond_res_block
            )

        # Output head to produce the final logits.
        out = []
        for i in range(n_out_res_block):
            out.append(GatedResBlock(channel, res_channel, 1))
        out.extend([nn.ELU(inplace=True), WNConv2d(channel, n_class, 1)])
        self.out = nn.Sequential(*out)

    def forward(self, input, condition=None, cache=None):
        if cache is None:
            cache = {} # Cache is used for efficient sequential generation.

        batch, height, width = input.shape
        # Convert integer input to one-hot vectors.
        input = (
            F.one_hot(input, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
        )

        # Initial horizontal and vertical streams.
        horizontal = shift_down(self.horizontal(input))
        vertical = shift_right(self.vertical(input))
        out = horizontal + vertical

        # Get the positional encodings for the current input size.
        background = self.background[:, :, :height, :].expand(batch, 2, height, width)

        # Process the conditional input if it exists.
        if condition is not None:
            # Use cached condition if available (for fast generation).
            if 'condition' in cache:
                condition = cache['condition']
                condition = condition[:, :, :height, :]
            else:
                # One-hot encode and process the condition through its ResNet.
                condition = (
                    F.one_hot(condition, self.n_class)
                    .permute(0, 3, 1, 2)
                    .type_as(self.background)
                )
                condition = self.cond_resnet(condition)
                # Upsample the condition to match the target resolution.
                condition = F.interpolate(condition, scale_factor=2)
                # Cache the processed condition.
                cache['condition'] = condition.detach().clone()
                condition = condition[:, :, :height, :]

        # Pass the features through the stack of PixelBlocks.
        for block in self.blocks:
            out = block(out, background, condition=condition)

        # Pass through the final output head to get logits.
        out = self.out(out)

        return out, cache