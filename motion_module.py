from enum import Enum
from typing import Optional

import math
import torch
from torch import nn
from einops import rearrange

import torch.nn as disable_weight_init
from ldm.modules.attention import FeedForward


class MotionModuleType(Enum):
    AnimateDiffV1 = "AnimateDiff V1, Yuwei Guo, Shanghai AI Lab"
    AnimateDiffV2 = "AnimateDiff V2, Yuwei Guo, Shanghai AI Lab"
    AnimateDiffV3 = "AnimateDiff V3, Yuwei Guo, Shanghai AI Lab"
    AnimateDiffXL = "AnimateDiff SDXL, Yuwei Guo, Shanghai AI Lab"
    SparseCtrl = "SparseCtrl, Yuwei Guo, Shanghai AI Lab"
    HotShotXL = "HotShot-XL, John Mullan, Natural Synthetics Inc"


    @staticmethod
    def get_mm_type(state_dict: dict[str, torch.Tensor]):
        keys = list(state_dict.keys())
        if any(["mid_block" in k for k in keys]):
            return MotionModuleType.AnimateDiffV2
        elif any(["down_blocks.3" in k for k in keys]):
            if 32 in next((state_dict[key] for key in state_dict if 'pe' in key), None).shape:
                return MotionModuleType.AnimateDiffV3
            else:
                return MotionModuleType.AnimateDiffV1
        else:
            if 32 in next((state_dict[key] for key in state_dict if 'pe' in key), None).shape:
                return MotionModuleType.AnimateDiffXL
            else:
                return MotionModuleType.HotShotXL


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


class MotionWrapper(nn.Module):
    def __init__(self, mm_name: str, mm_hash: str, mm_type: MotionModuleType, operations = disable_weight_init):
        super().__init__()
        self.mm_name = mm_name
        self.mm_type = mm_type
        self.mm_hash = mm_hash
        max_len = 24 if self.enable_gn_hack() else 32
        in_channels = (320, 640, 1280) if self.is_xl else (320, 640, 1280, 1280)
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        for c in in_channels:
            if mm_type in [MotionModuleType.SparseCtrl]:
                self.down_blocks.append(MotionModule(c, num_mm=2, max_len=max_len, attention_block_types=("Temporal_Self", ), operations=operations))
            else:
                self.down_blocks.append(MotionModule(c, num_mm=2, max_len=max_len, operations=operations))
                self.up_blocks.insert(0,MotionModule(c, num_mm=3, max_len=max_len, operations=operations))
        if mm_type in [MotionModuleType.AnimateDiffV2]:
            self.mid_block = MotionModule(1280, num_mm=1, max_len=max_len, operations=operations)


    def enable_gn_hack(self):
        return self.mm_type in [MotionModuleType.AnimateDiffV1, MotionModuleType.HotShotXL]


    @property
    def is_xl(self):
        return self.mm_type in [MotionModuleType.AnimateDiffXL, MotionModuleType.HotShotXL]


    @property
    def is_adxl(self):
        return self.mm_type == MotionModuleType.AnimateDiffXL

    @property
    def is_hotshot(self):
        return self.mm_type == MotionModuleType.HotShotXL


    @property
    def is_v2(self):
        return self.mm_type == MotionModuleType.AnimateDiffV2


class MotionModule(nn.Module):
    def __init__(self, in_channels, num_mm, max_len, attention_block_types=("Temporal_Self", "Temporal_Self"), operations = disable_weight_init):
        super().__init__()
        self.motion_modules = nn.ModuleList([
            VanillaTemporalModule(
                in_channels=in_channels,
                temporal_position_encoding_max_len=max_len,
                attention_block_types=attention_block_types,
                operations=operations,)
            for _ in range(num_mm)])


    def forward(self, x: torch.Tensor):
        for mm in self.motion_modules:
            x = mm(x)
        return x


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads                = 8,
        num_transformer_block              = 1,
        attention_block_types              =( "Temporal_Self", "Temporal_Self" ),
        temporal_position_encoding_max_len = 24,
        temporal_attention_dim_div         = 1,
        zero_initialize                    = True,
        operations                         = disable_weight_init,
    ):
        super().__init__()
        
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
            operations=operations,
        )
        
        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)


    def forward(self, x: torch.Tensor):
        return self.temporal_transformer(x)


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,

        num_layers,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),        
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        
        temporal_position_encoding_max_len = 24,

        operations                         = disable_weight_init,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = operations.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = operations.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    operations=operations,
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = operations.Linear(inner_dim, in_channels)    
    
    def forward(self, hidden_states: torch.Tensor):
        _, _, height, _ = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states).type(hidden_states.dtype)
        hidden_states = rearrange(hidden_states, "b c h w -> b (h w) c")
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        
        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = rearrange(hidden_states, "b (h w) c -> b c h w", h=height)

        output = hidden_states + residual
        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),
        dropout                            = 0.0,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        temporal_position_encoding_max_len = 24,
        operations                         = disable_weight_init,
    ):
        super().__init__()

        attention_blocks = []
        norms = []
        
        for _ in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    operations=operations,
                )
            )
            norms.append(operations.LayerNorm(dim))
            
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, glu=(activation_fn=='geglu'))
        self.ff_norm = operations.LayerNorm(dim)


    def forward(self, hidden_states: torch.Tensor):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states).type(hidden_states.dtype)
            hidden_states = attention_block(norm_hidden_states) + hidden_states
            
        hidden_states = self.ff(self.ff_norm(hidden_states).type(hidden_states.dtype)) + hidden_states
        
        output = hidden_states  
        return output


class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_len = 24,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x)
        return self.dropout(x)


class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        operations = disable_weight_init,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = operations.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = operations.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = operations.Linear(cross_attention_dim, inner_dim, bias=bias)

        self.to_out = nn.Sequential(operations.Linear(inner_dim, query_dim), nn.Dropout(dropout))


class VersatileAttention(CrossAttention):
    def __init__(
            self,
            temporal_position_encoding_max_len = 24,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"], 
            max_len=temporal_position_encoding_max_len)


    def forward(self, x: torch.Tensor):
        from scripts.animatediff_mm import mm_animatediff
        video_length = mm_animatediff.ad_params.batch_size

        d = x.shape[1]
        x = rearrange(x, "(b f) d c -> (b d) f c", f=video_length)
        x = self.pos_encoder(x)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b s (h d) -> (b h) s d', h=self.heads), (q, k, v))
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, '(b h) s d -> b s (h d)', h=self.heads)

        x = self.to_out(x) # linear proj and dropout
        x = rearrange(x, "(b d) f c -> (b f) d c", d=d)

        return x
