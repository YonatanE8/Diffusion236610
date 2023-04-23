# Implementation is heavily based on https://github.com/labmlai/annotated_deep_learning_paper_implementations
# with minor edits to make it a 1D model
from typing import List, Optional, Union, Dict
from Diffusion236610.models.fc_models import get_fc_layer
from Diffusion236610.utils.defaults import MODELS_TENSOR_PREDICITONS_KEY

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalization(channels: int):
    """
    Group normalization
    This is a helper function, with fixed number of groups..
    """

    return GroupNorm32(
        num_groups=32,
        num_channels=channels,
    )


class TemporalTransformer(nn.Module):
    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int, use_dilation: bool = True):
        """
        :param channels: is the number of channels in the feature map
        :param n_heads: is the number of attention heads
        :param n_layers: is the number of transformer layers
        :param d_cond: is the size of the conditional embedding
        :param use_dilation: Whether to replace strides with dilation in the convolutional layers
        """

        super().__init__()
        # Initial group normalization
        self.norm = nn.GroupNorm(
            num_groups=32,
            num_channels=channels,
            eps=1e-6,
            affine=True,
        )

        # Initial convolution
        if use_dilation:
            self.proj_in = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                dilation=1,
                padding=0,
            )

        else:
            self.proj_in = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    d_model=channels,
                    n_heads=n_heads,
                    d_head=(channels // n_heads),
                    d_cond=d_cond,
                )
                for _ in range(n_layers)
            ]
        )

        # Final convolution
        if use_dilation:
            self.proj_out = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                dilation=1,
                padding=0,
            )

        else:
            self.proj_out = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: is the feature map of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """

        # Get shape `[batch_size, channels, length]`
        batch_size, channels, length = x.shape

        # For residual connection
        x_in = x

        # Normalize
        x = self.norm(x)

        # Initial convolution
        x = self.proj_in(x)

        # Transpose and reshape from `[batch_size, length]`
        # to `[batch_size, length, channels]`
        x = x.permute(0, 2, 1).view(batch_size, length, channels)

        # Apply the transformer layers
        for block in self.transformer_blocks:
            x = block(x, cond)

        # Reshape and transpose from `[batch_size, length, channels]`
        # to `[batch_size, channels, length]`
        x = x.view(batch_size, length, channels).permute(0, 2, 1)

        # Final convolution
        x = self.proj_out(x)

        # Add residual
        return x + x_in

    def __call__(self, x: torch.Tensor, cond: torch.Tensor):
        return self.forward(x=x, cond=cond)


class BasicTransformerBlock(nn.Module):
    """
    Basic Transformer Layer
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        """

        super().__init__()

        # Self-attention layer and pre-norm layer
        self.attn1 = CrossAttention(d_model, d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention layer and pre-norm layer
        self.attn2 = CrossAttention(d_model, d_cond, n_heads, d_head)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network and pre-norm layer
        self.ff = FeedForward(d_model=d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """

        # Self attention
        x = self.attn1(self.norm1(x)) + x

        # Cross-attention with conditioning
        x = self.attn2(self.norm2(x), cond=cond) + x

        # Feed-forward network
        x = self.ff(self.norm3(x)) + x

        return x

    def __call__(self, x: torch.Tensor, cond: torch.Tensor):
        return self.forward(x=x, cond=cond)


class SelfAttentionConditioningModule(nn.Module):
    """
    Self-Attention based module for processing the initial conditioning inputs
    """

    def __init__(
            self,
            input_dim: int,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            bias: bool = True,
            activation: Union[str, Dict] = 'leakyrelu'
    ):
        super().__init__()

        self.values = get_fc_layer(
            input_dim=input_dim,
            output_dim=embed_dim,
            bias=bias,
            activation=activation,
            dropout=dropout,
        )
        self.keys = get_fc_layer(
            input_dim=input_dim,
            output_dim=embed_dim,
            bias=bias,
            activation=activation,
            dropout=dropout,
        )
        self.queries = get_fc_layer(
            input_dim=input_dim,
            output_dim=embed_dim,
            bias=bias,
            activation=activation,
            dropout=dropout,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )

        self.gegelu = GeGLU(
            d_in=embed_dim,
            d_out=embed_dim,
        )

    def __call__(self, x: torch.Tensor):
        return self.forward(x=x)

    def forward(self, x: torch.Tensor):
        v = self.values(x)
        k = self.keys(x)
        q = self.queries(x)
        sa = self.attention.forward(
            query=q,
            value=v,
            key=k,
            need_weights=False,
        )
        cond = self.gegelu(sa[0])

        return cond


class CrossAttention(nn.Module):
    """
    Cross Attention Layer
    This falls-back to self-attention when conditional embeddings are not specified.
    """

    def __init__(
            self,
            d_model: int,
            d_cond: int,
            n_heads: int,
            d_head: int,
            is_inplace: bool = False,
            use_flash_attention: bool = False
    ):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        :param use_flash_attention: Whether to use Flash Attention for the Self/Cross Attention computations.
        Requires installing the flash attention package (`pip install flash-attn`),
        which in turns requires CUDA 11, NVCC, and a Turing/Ampere GPU.
        """

        super().__init__()

        self.is_inplace = is_inplace
        self.n_heads = n_heads
        self.d_head = d_head
        self.use_flash_attention = use_flash_attention

        # Attention scaling factor
        self.scale = d_head ** -0.5

        # Query, key and value mappings
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(
            in_features=d_model,
            out_features=d_attn,
            bias=False,
        )
        self.to_k = nn.Linear(
            in_features=d_cond,
            out_features=d_attn,
            bias=False,
        )
        self.to_v = nn.Linear(
            in_features=d_cond,
            out_features=d_attn,
            bias=False,
        )

        # Final linear layer
        self.to_out = nn.Sequential(
            nn.Linear(
                in_features=d_attn,
                out_features=d_model,
            )
        )

        # Setup [flash attention](https://github.com/HazyResearch/flash-attention).
        # Flash attention is only used if it's installed
        # and `CrossAttention.use_flash_attention` is set to `True`.
        if use_flash_attention:
            from flash_attn.flash_attention import FlashAttention
            self.flash = FlashAttention()

            # Set the scale for scaled dot-product attention.
            self.flash.softmax_scale = self.scale

        # Set to `None` if it's not installed
        else:
            self.flash = None

    def __call__(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        return self.forward(x=x, cond=cond)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond]`
        """

        # If `cond` is `None` we perform self attention
        has_cond = cond is not None
        if not has_cond:
            cond = x

        # Get query, key and value vectors
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        # Use flash attention if it's available and the head size is less than or equal to `128`
        if self.use_flash_attention and self.flash is not None and not has_cond and self.d_head <= 128:
            return self.flash_attention(q, k, v)

        # Otherwise, fallback to normal attention
        else:
            return self.normal_attention(q, k, v)

    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Flash Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Get batch size and number of elements along sequence axis (`width * height`)
        batch_size, seq_len, _ = q.shape

        # Stack `q`, `k`, `v` vectors for flash attention, to get a single tensor of
        # shape `[batch_size, seq_len, 3, n_heads * d_head]`
        qkv = torch.stack((q, k, v), dim=2)

        # Split the heads
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)

        # Flash attention works for head sizes `32`, `64` and `128`, so we have to pad the heads to
        # fit this size.
        if self.d_head <= 32:
            pad = 32 - self.d_head

        elif self.d_head <= 64:
            pad = 64 - self.d_head

        elif self.d_head <= 128:
            pad = 128 - self.d_head

        else:
            raise ValueError(f'Head size ${self.d_head} too large for Flash Attention')

        # Pad the heads
        if pad:
            qkv = torch.cat((qkv, qkv.new_zeros(batch_size, seq_len, 3, self.n_heads, pad)), dim=-1)

        # Compute attention
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        # This gives a tensor of shape `[batch_size, seq_len, n_heads, d_padded]`
        out, _ = self.flash(qkv)

        # Truncate the extra head size
        out = out[..., :self.d_head]

        # Reshape to `[batch_size, seq_len, n_heads * d_head]`
        out = out.reshape(batch_size, seq_len, (self.n_heads * self.d_head))

        # Map to `[batch_size, length, d_model]` with a linear layer
        return self.to_out(out)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Normal Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Split them to heads of shape `[batch_size, seq_len, n_heads, d_head]`
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)

        # Calculate attention $\frac{Q K^\top}{\sqrt{d_{key}}}$
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale

        # Compute softmax
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$$
        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)

        else:
            attn = attn.softmax(dim=-1)

        # Compute attention output
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        out = torch.einsum('bhij,bjhd->bihd', attn, v)

        # Reshape to `[batch_size, length, n_heads * d_head]`
        out = out.reshape(*out.shape[:2], -1)

        # Map to `[batch_size, length, d_model]` with a linear layer
        return self.to_out(out)


class FeedForward(nn.Module):
    """
    Feed-Forward Network
    """

    def __init__(
            self,
            d_model: int,
            d_mult: int = 4,
            bias: bool = True,
            dropout: float = 0.0,
    ):
        """
        :param d_model: is the input embedding size
        :param d_mult: is multiplicative factor for the hidden layer size
        :param bias: Whether to add a bias parameter in the Linear layers
        :param dropout: dropout rate.
        """

        super().__init__()
        self.net = nn.Sequential(
            GeGLU(
                d_in=d_model,
                d_out=(d_model * d_mult),
            ),
            nn.Dropout(dropout),
            nn.Linear(
                in_features=(d_model * d_mult),
                out_features=d_model,
                bias=bias,
            )
        )

    def __call__(self, x: torch.Tensor):
        return self.forward(x=x)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class GeGLU(nn.Module):
    """
    GeGLU Activation
    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """

    def __init__(
            self,
            d_in: int,
            d_out: int,
    ):
        super().__init__()

        # Combined linear projections $xW + b$ and $xV + c$
        self.proj = nn.Linear(d_in, d_out * 2)

    def __call__(self, x: torch.Tensor):
        return self.forward(x=x)

    def forward(self, x: torch.Tensor):
        # Get $xW + b$ and $xV + c$
        x, gate = self.proj(x).chunk(2, dim=-1)

        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return x * F.gelu(gate)


class TimestepEmbedSequential(nn.Sequential):
    """
    Sequential block for modules with different inputs
    This sequential module can compose of different modules suck as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond: Optional[torch.Tensor] = None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)

            elif isinstance(layer, TemporalTransformer):
                x = layer(x, cond)

            else:
                x = layer(x)

        return x

    def __call__(self, x: torch.Tensor, t_emb: torch.Tensor, cond: Optional[torch.Tensor] = None):
        return self.forward(x=x, t_emb=t_emb, cond=cond)


class UpSample(nn.Module):
    """
    Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """

        super().__init__()

        # convolution mapping
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, length]`
        """

        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        # Apply convolution
        return self.conv(x)

    def __call__(self, x: torch.Tensor):
        return self.forward(x=x)


class DownSample(nn.Module):
    """
    Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """

        super().__init__()

        # convolution with stride length of $2$ to down-sample by a factor of $2$
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, length]`
        """

        # Apply convolution
        return self.conv(x)

    def __call__(self, x: torch.Tensor):
        return self.forward(x=x)


class ResBlock(nn.Module):
    """
    ResNet Block
    """

    def __init__(
            self,
            channels: int,
            d_t_emb: int,
            *,
            out_channels: int = None,
            bias: bool = True,
            dropout: float = 0.0,
    ):
        """
        :param channels: the number of input channels
        :param d_t_emb: the size of timestep embeddings
        :param out_channels: is the number of out channels. defaults to `channels.
        :param bias: Whether to add a bias parameter in the Linear layers
        :param dropout: dropout rate.
        """

        super().__init__()

        # `out_channels` not specified
        if out_channels is None:
            out_channels = channels

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        # Time step embeddings
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                in_features=d_t_emb,
                out_features=out_channels,
                bias=bias,
            ),
        )

        # Final convolution layer
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
        )

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == channels:
            self.skip_connection = nn.Identity()

        else:
            self.skip_connection = nn.Conv1d(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def __call__(self, x: torch.Tensor, t_emb: torch.Tensor):
        return self.forward(x=x, t_emb=t_emb)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, length]`
        :param t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`
        """

        # Initial convolution
        h = self.in_layers(x)

        # Time step embeddings
        t_emb = self.emb_layers(t_emb).type(h.dtype)

        # Add time step embeddings
        h = h + t_emb[:, :, None]

        # Final convolution
        h = self.out_layers(h)

        # Add skip connection
        return self.skip_connection(x) + h


class GroupNorm32(nn.GroupNorm):
    """
    Group normalization with float32 casting
    """

    def __call__(self, x: torch.Tensor):
        return self.forward(x=x)

    def forward(self, x: torch.Tensor):
        return super().forward(x.float()).type(x.dtype)


class UNet1D(nn.Module):
    """
    1D U-Net model
    """

    def __init__(
            self,
            *,
            in_channels: int,
            out_channels: int,
            channels: int,
            n_res_blocks: int,
            channel_multipliers: List[int],
            n_heads: int,
            attention_levels: Optional[List[int]] = (),
            tf_layers: int = 0,
            d_cond: int = 768,
            use_dilation: bool = True,
            bias: bool = True,
            dropout: float = 0.0,
    ):
        """
        :param in_channels: is the number of channels in the input feature map
        :param out_channels: is the number of channels in the output feature map
        :param channels: is the base channel count for the model
        :param n_res_blocks: number of residual blocks at each level
        :param attention_levels: are the levels at which attention should be performed
        :param channel_multipliers: are the multiplicative factors for number of channels for each level
        :param n_heads: the number of attention heads in the transformers
        :param use_dilation: Whether to replace strides with dilation in the convolutional layers
        :param bias: Whether to add a bias parameter in the Linear layers
        :param dropout: dropout rate.
        """

        super().__init__()
        self.channels = channels

        # Number of levels
        levels = len(channel_multipliers)

        # Size time embeddings
        d_time_emb = channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(
                in_features=channels,
                out_features=d_time_emb,
                bias=bias,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=d_time_emb,
                out_features=d_time_emb,
                bias=bias,
            ),
        )

        # Input half of the U-Net
        self.input_blocks = nn.ModuleList()

        # Initial convolution that maps the input to `channels`.
        # The blocks are wrapped in `TimestepEmbedSequential` module because
        # different modules have different forward function signatures;
        # for example, convolution only accepts the feature map and
        # residual blocks accept the feature map and time embedding.
        # `TimestepEmbedSequential` calls them accordingly.
        self.input_blocks.append(
            TimestepEmbedSequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=3,
                    padding=1,
                )
            )
        )

        # Number of channels at each block in the input half of U-Net
        input_block_channels = [channels]

        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]

        # Prepare levels
        for i in range(levels):
            # Add the residual blocks and attentions
            for _ in range(n_res_blocks):
                # Residual block maps from previous number of channels to the number of
                # channels in the current level
                layers = [
                    ResBlock(
                        channels=channels,
                        d_t_emb=d_time_emb,
                        out_channels=channels_list[i],
                        bias=bias,
                        dropout=dropout,
                    )
                ]
                channels = channels_list[i]

                # Add transformer
                if i in attention_levels:
                    layers.append(
                        TemporalTransformer(
                            channels=channels,
                            n_heads=n_heads,
                            n_layers=tf_layers,
                            d_cond=d_cond,
                            use_dilation=use_dilation,
                        )
                    )

                # Add them to the input half of the U-Net and keep track of the number of channels of
                # its output
                self.input_blocks.append(
                    TimestepEmbedSequential(*layers)
                )
                input_block_channels.append(channels)

            # Down sample at all levels except last
            if i != levels - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        DownSample(channels=channels)
                    )
                )
                input_block_channels.append(channels)

        # The middle of the U-Net
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=channels,
                d_t_emb=d_time_emb,
                bias=bias,
                dropout=dropout,
            ),
            TemporalTransformer(
                channels=channels,
                n_heads=n_heads,
                n_layers=tf_layers,
                d_cond=d_cond,
                use_dilation=use_dilation,
            ),
            ResBlock(
                channels=channels,
                d_t_emb=d_time_emb,
                bias=bias,
                dropout=dropout,
            ),
        )

        # Second half of the U-Net
        self.output_blocks = nn.ModuleList([])

        # Prepare levels in reverse order
        for i in reversed(range(levels)):
            # Add the residual blocks and attentions
            for j in range(n_res_blocks + 1):
                # Residual block maps from previous number of channels plus the
                # skip connections from the input half of U-Net to the number of
                # channels in the current level.
                layers = [
                    ResBlock(
                        channels=(channels + input_block_channels.pop()),
                        d_t_emb=d_time_emb,
                        out_channels=channels_list[i],
                        bias=bias,
                        dropout=dropout,
                    )
                ]
                channels = channels_list[i]

                # Add transformer
                if i in attention_levels:
                    layers.append(
                        TemporalTransformer(
                            channels=channels,
                            n_heads=n_heads,
                            n_layers=tf_layers,
                            d_cond=d_cond,
                            use_dilation=use_dilation,
                        )
                    )

                # Up-sample at every level after last residual block
                # except the last one.
                # Note that we are iterating in reverse; i.e. `i == 0` is the last.
                if i != 0 and j == n_res_blocks:
                    layers.append(
                        UpSample(channels=channels)
                    )

                # Add to the output half of the U-Net
                self.output_blocks.append(
                    TimestepEmbedSequential(*layers)
                )

        # Final normalization and $3 \times 3$ convolution
        self.output_layer = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings
        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """

        # $\frac{c}{2}$; half the channels are sin and the other half is cos,
        half = self.channels // 2

        # $\frac{1}{10000^{\frac{2i}{c}}}$
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)

        # $\frac{t}{10000^{\frac{2i}{c}}}$
        args = time_steps[:, None].float() * frequencies[None]

        # $\cos\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$ and $\sin\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def __call__(self, x: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor):
        return self.forward(x=x, time_steps=time_steps, cond=cond)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        :param time_steps: are the time steps of shape `[batch_size]`
        :param cond: conditioning of shape `[batch_size, n_cond, d_cond]`
        """

        # To store the input half outputs for skip connections
        x_input_block = []

        # Get time step embeddings
        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embed(t_emb)

        # Input half of the U-Net
        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input_block.append(x)

        # Middle of the U-Net
        x = self.middle_block(x, t_emb, cond)

        # Output half of the U-Net
        for i, module in enumerate(self.output_blocks):
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond)

        # Final normalization and $3 \times 3$ convolution
        out = self.output_layer(x)

        out = {
            MODELS_TENSOR_PREDICITONS_KEY: out,
        }

        return out


class FixedUNet1D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            channels_multiplication_factor: int = 2,
            n_transformers_layers: int = 3,
            n_heads: int = 8,
            d_cond: int = 768,
            use_dilation: bool = True,
            bias: bool = True,
            dropout: float = 0.0,
    ):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._channels = channels
        self._channels_multiplication_factor = channels_multiplication_factor
        self._n_transformers_layers = n_transformers_layers
        self._n_heads = n_heads
        self._d_cond = d_cond
        self._use_dilation = use_dilation
        self._bias = bias
        self._dropout = dropout

        # Size time embeddings
        d_time_emb = channels * 4
        self._time_embed = nn.Sequential(
            nn.Linear(
                in_features=channels,
                out_features=d_time_emb,
                bias=bias,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=d_time_emb,
                out_features=d_time_emb,
                bias=bias,
            ),
        )

        # Input half of the U-Net
        self._out_channels_per_input_block = [channels, ]
        self._in_block_1 = TimestepEmbedSequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
            )
        )

        self._out_channels_per_input_block.append(channels)
        self._in_block_2 = TimestepEmbedSequential(
            ResBlock(
                channels=channels,
                d_t_emb=d_time_emb,
                out_channels=channels,
                bias=bias,
                dropout=dropout,
            ),
            TemporalTransformer(
                channels=channels,
                n_heads=n_heads,
                n_layers=n_transformers_layers,
                d_cond=d_cond,
                use_dilation=use_dilation,
            ),
            ResBlock(
                channels=channels,
                d_t_emb=d_time_emb,
                out_channels=channels,
                bias=bias,
                dropout=dropout,
            ),
            DownSample(channels=channels),
        )

        out_channels = channels_multiplication_factor * channels
        self._out_channels_per_input_block.append(out_channels)
        self._in_block_3 = TimestepEmbedSequential(
            ResBlock(
                channels=channels,
                d_t_emb=d_time_emb,
                out_channels=out_channels,
                bias=bias,
                dropout=dropout,
            ),
            TemporalTransformer(
                channels=out_channels,
                n_heads=n_heads,
                n_layers=n_transformers_layers,
                d_cond=d_cond,
                use_dilation=use_dilation,
            ),
            ResBlock(
                channels=out_channels,
                d_t_emb=d_time_emb,
                out_channels=out_channels,
                bias=bias,
                dropout=dropout,
            ),
            DownSample(channels=out_channels),
        )

        channels = out_channels
        out_channels = channels_multiplication_factor * channels
        self._out_channels_per_input_block.append(out_channels)
        self._in_block_4 = TimestepEmbedSequential(
            ResBlock(
                channels=channels,
                d_t_emb=d_time_emb,
                out_channels=out_channels,
                bias=bias,
                dropout=dropout,
            ),
            TemporalTransformer(
                channels=out_channels,
                n_heads=n_heads,
                n_layers=n_transformers_layers,
                d_cond=d_cond,
                use_dilation=use_dilation,
            ),
            ResBlock(
                channels=out_channels,
                d_t_emb=d_time_emb,
                out_channels=out_channels,
                bias=bias,
                dropout=dropout,
            ),
            DownSample(channels=out_channels),
        )
        channels = out_channels
        out_channels = channels_multiplication_factor * channels
        self._out_channels_per_input_block.append(out_channels)
        self._in_block_5 = TimestepEmbedSequential(
            ResBlock(
                channels=channels,
                d_t_emb=d_time_emb,
                out_channels=out_channels,
                bias=bias,
                dropout=dropout,
            ),
            TemporalTransformer(
                channels=out_channels,
                n_heads=n_heads,
                n_layers=n_transformers_layers,
                d_cond=d_cond,
                use_dilation=use_dilation,
            ),
            ResBlock(
                channels=out_channels,
                d_t_emb=d_time_emb,
                out_channels=out_channels,
                bias=bias,
                dropout=dropout,
            ),
        )

        # The middle of the U-Net
        channels = out_channels
        self._middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=channels,
                out_channels=channels,
                d_t_emb=d_time_emb,
                bias=bias,
                dropout=dropout,
            ),
            TemporalTransformer(
                channels=channels,
                n_heads=n_heads,
                n_layers=n_transformers_layers,
                d_cond=d_cond,
                use_dilation=use_dilation,
            ),
            ResBlock(
                channels=channels,
                out_channels=channels,
                d_t_emb=d_time_emb,
                bias=bias,
                dropout=dropout,
            ),
        )

        # Second half of the U-Net
        self._out_block_1 = TimestepEmbedSequential(
            ResBlock(
                channels=(channels + self._out_channels_per_input_block[-1]),
                d_t_emb=d_time_emb,
                out_channels=self._out_channels_per_input_block[-1],
                bias=bias,
                dropout=dropout,
            ),
            TemporalTransformer(
                channels=self._out_channels_per_input_block[-1],
                n_heads=n_heads,
                n_layers=n_transformers_layers,
                d_cond=d_cond,
                use_dilation=use_dilation,
            ),
            ResBlock(
                channels=self._out_channels_per_input_block[-1],
                d_t_emb=d_time_emb,
                out_channels=self._out_channels_per_input_block[-1],
                bias=bias,
                dropout=dropout,
            ),
        )

        channels = self._out_channels_per_input_block[-1]
        self._out_block_2 = TimestepEmbedSequential(
            ResBlock(
                channels=(channels + self._out_channels_per_input_block[-2]),
                d_t_emb=d_time_emb,
                out_channels=self._out_channels_per_input_block[-2],
                bias=bias,
                dropout=dropout,
            ),
            TemporalTransformer(
                channels=self._out_channels_per_input_block[-2],
                n_heads=n_heads,
                n_layers=n_transformers_layers,
                d_cond=d_cond,
                use_dilation=use_dilation,
            ),
            ResBlock(
                channels=self._out_channels_per_input_block[-2],
                d_t_emb=d_time_emb,
                out_channels=self._out_channels_per_input_block[-2],
                bias=bias,
                dropout=dropout,
            ),
        )
        self._upsample_1 = UpSample(channels=self._out_channels_per_input_block[-2])

        channels = self._out_channels_per_input_block[-2]
        self._out_block_3 = TimestepEmbedSequential(
            ResBlock(
                channels=(channels + self._out_channels_per_input_block[-3]),
                d_t_emb=d_time_emb,
                out_channels=self._out_channels_per_input_block[-3],
                bias=bias,
                dropout=dropout,
            ),
            TemporalTransformer(
                channels=self._out_channels_per_input_block[-3],
                n_heads=n_heads,
                n_layers=n_transformers_layers,
                d_cond=d_cond,
                use_dilation=use_dilation,
            ),
            ResBlock(
                channels=self._out_channels_per_input_block[-3],
                d_t_emb=d_time_emb,
                out_channels=self._out_channels_per_input_block[-3],
                bias=bias,
                dropout=dropout,
            ),
        )
        self._upsample_2 = UpSample(channels=self._out_channels_per_input_block[-3])

        channels = self._out_channels_per_input_block[-3]
        self._out_block_4 = TimestepEmbedSequential(
            ResBlock(
                channels=(channels + self._out_channels_per_input_block[-4]),
                d_t_emb=d_time_emb,
                out_channels=self._out_channels_per_input_block[-4],
                bias=bias,
                dropout=dropout,
            ),
            TemporalTransformer(
                channels=self._out_channels_per_input_block[-4],
                n_heads=n_heads,
                n_layers=n_transformers_layers,
                d_cond=d_cond,
                use_dilation=use_dilation,
            ),
            ResBlock(
                channels=self._out_channels_per_input_block[-4],
                d_t_emb=d_time_emb,
                out_channels=self._out_channels_per_input_block[-4],
                bias=bias,
                dropout=dropout,
            ),
        )
        self._upsample_3 = UpSample(channels=self._out_channels_per_input_block[-4])

        channels = self._out_channels_per_input_block[-4]
        self._out_block_5 = TimestepEmbedSequential(
            ResBlock(
                channels=(channels + self._out_channels_per_input_block[-5]),
                d_t_emb=d_time_emb,
                out_channels=self._out_channels_per_input_block[-5],
                bias=bias,
                dropout=dropout,
            ),
            TemporalTransformer(
                channels=self._out_channels_per_input_block[-5],
                n_heads=n_heads,
                n_layers=n_transformers_layers,
                d_cond=d_cond,
                use_dilation=use_dilation,
            ),
            ResBlock(
                channels=self._out_channels_per_input_block[-5],
                d_t_emb=d_time_emb,
                out_channels=self._out_channels_per_input_block[-5],
                bias=bias,
                dropout=dropout,
            ),
        )

        # Final normalization and $3 \times 3$ convolution
        channels = self._out_channels_per_input_block[-5]
        self.output_layer = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(
                in_channels=channels,
                out_channels=self._out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings
        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """

        # $\frac{c}{2}$; half the channels are sin and the other half is cos,
        half = self._channels // 2

        # $\frac{1}{10000^{\frac{2i}{c}}}$
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)

        # $\frac{t}{10000^{\frac{2i}{c}}}$
        args = time_steps[:, None].float() * frequencies[None]

        # $\cos\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$ and $\sin\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def __call__(self, x: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor):
        return self.forward(x=x, time_steps=time_steps, cond=cond)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        :param time_steps: are the time steps of shape `[batch_size]`
        :param cond: conditioning of shape `[batch_size, n_cond, d_cond]`
        """

        # Get time step embeddings
        t_emb = self.time_step_embedding(time_steps)
        t_emb = self._time_embed(t_emb)

        # Input half of the U-Net
        in_1 = self._in_block_1(x, t_emb, cond)
        in_2 = self._in_block_2(in_1, t_emb, cond)
        in_3 = self._in_block_3(in_2, t_emb, cond)
        in_4 = self._in_block_4(in_3, t_emb, cond)
        in_5 = self._in_block_5(in_4, t_emb, cond)

        # Middle of the U-Net
        middle_out = self._middle_block(in_5, t_emb, cond)

        # Output half of the U-Net
        out = self._out_block_1(
            torch.cat([middle_out, in_5], dim=1),
            t_emb,
            cond,
        )
        out = self._out_block_2(
            torch.cat([out, in_4], dim=1),
            t_emb,
            cond,
        )
        out = self._upsample_1(out)
        out = self._out_block_3(
            torch.cat([out, in_3], dim=1),
            t_emb,
            cond,
        )
        out = self._upsample_2(out)
        out = self._out_block_4(
            torch.cat([out, in_2], dim=1),
            t_emb,
            cond,
        )
        out = self._upsample_3(out)
        out = self._out_block_5(
            torch.cat([out, in_1], dim=1),
            t_emb,
            cond,
        )

        # Final normalization and $3 \times 3$ convolution
        out = self.output_layer(out)

        out = {
            MODELS_TENSOR_PREDICITONS_KEY: out,
        }

        return out
