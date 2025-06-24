# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch GeminiMla model, incorporating novel architectural features. """
import math
import warnings
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.generation import GenerationMixin
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    ModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.configuration_utils import PretrainedConfig
import torch.distributed as dist
import numpy as np

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GeminiMlaConfig"
# --- Configuration Class ---
class GeminiMlaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GeminiMlaModel`]. It is used to instantiate a
    Gemini-MLA model according to the specified arguments, defining the model architecture.
    """

    model_type = "gemini_mla"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size = 2048,
        num_hidden_layers=61,
        num_nextn_predict_layers=1,
        num_attention_heads=128,
        num_key_value_heads=128,
        n_shared_experts = 1,
        n_routed_experts = 256,
        ep_size = 1,
        routed_scaling_factor = 2.5,
        kv_lora_rank = 512,
        q_lora_rank = 1536,
        qk_rope_head_dim = 64,
        v_head_dim = 128,
        qk_nope_head_dim = 128,
        topk_method = 'noaux_tc',
        n_group = 8,
        topk_group = 4,
        num_experts_per_tok = 8,
        moe_layer_freq = 1,
        first_k_dense_replace = 3,
        norm_topk_prob = True,
        scoring_func = 'sigmoid',
        aux_loss_alpha = 0.001,
        mtp_beta = 0.1,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        # --- New args for Gemini-MLA features ---
        use_scaled_embeddings: bool = False,
        embedding_scale_factor: float = 0.00848388671875,
        use_laurel_residual: bool = False,
        laurel_hidden_dim: int = 64,
        use_per_layer_embeddings: bool = False,
        per_layer_embedding_dim: int = 256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.mtp_beta = mtp_beta
        
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # --- New Gemini-MLA feature assignments ---
        self.use_scaled_embeddings = use_scaled_embeddings
        self.embedding_scale_factor = embedding_scale_factor
        self.use_laurel_residual = use_laurel_residual
        self.laurel_hidden_dim = laurel_hidden_dim
        self.use_per_layer_embeddings = use_per_layer_embeddings
        self.per_layer_embedding_dim = per_layer_embedding_dim

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
    
# --- New Custom Output Data Classes ---
@dataclass
class BaseModelOutputWithPastAndAuxLoss(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    aux_losses: Optional[List[torch.FloatTensor]] = None


@dataclass
class CausalLMOutputWithPastAndAuxLoss(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    aux_losses: Optional[List[torch.FloatTensor]] = None


# --- Start of Model Architecture ---

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class GeminiMlaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        GeminiMlaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(GeminiMlaRMSNorm)

# --- START: New Architectural Modules for Gemini-MLA ---

class GeminiMlaScaledEmbedder(nn.Embedding):
    """
    An embedding layer that scales the output embeddings by a fixed factor.
    It inherits from nn.Embedding to be compatible with Hugging Face's weight tying mechanism.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, scale_factor: float, padding_idx: Optional[int] = None):
        super().__init__(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.scale_factor = scale_factor

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        # The `super().forward()` call will execute the original nn.Embedding's forward pass.
        embeddings = super().forward(token_ids)
        return embeddings * self.scale_factor


class GeminiMlaLaurelBlock(nn.Module):
    """
    An enhanced residual connection block using a low-rank transformation (Laurel).
    It modifies an input tensor `x` to `x + Norm(Linear_2(Linear_1(x)))`.
    """
    def __init__(self, config: GeminiMlaConfig):
        super().__init__()
        self.linear_left = nn.Linear(config.hidden_size, config.laurel_hidden_dim, bias=False)
        self.linear_right = nn.Linear(config.laurel_hidden_dim, config.hidden_size, bias=False)
        self.norm = GeminiMlaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Low-rank transformation
        transformed = self.linear_right(self.linear_left(x))
        normalized = self.norm(transformed)
        return x + normalized


class GeminiMlaPerLayerEmbedder(nn.Module):
    """
    Creates per-layer token embeddings. For each token in the input, it generates
    a unique embedding vector for each of the N transformer layers.
    """
    def __init__(self, config: GeminiMlaConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.per_layer_embedding_dim
        self.num_layers = config.num_hidden_layers
        self.padding_idx = config.pad_token_id

        self.embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, self.embedding_dim, self.padding_idx)
            for _ in range(self.num_layers)
        ])

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        layer_embeddings = []
        for embedding_layer in self.embeddings:
            layer_embeddings.append(embedding_layer(token_ids))
        # Stack along a new dimension for layers.
        # Output shape: (batch, seq_len, num_layers, per_layer_embedding_dim)
        per_layer_embeddings = torch.stack(layer_embeddings, dim=2)
        return per_layer_embeddings


class GeminiMlaPerLayerFusion(nn.Module):
    """
    Fuses the per-layer embedding with the main hidden state at the end of a transformer layer.
    """
    def __init__(self, config: GeminiMlaConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.per_layer_embedding_dim, bias=False)
        self.proj = nn.Linear(config.per_layer_embedding_dim, config.hidden_size, bias=False)
        self.norm = GeminiMlaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, per_layer_embedding: torch.Tensor) -> torch.Tensor:
        # per_layer_embedding is already selected for the current layer, shape (B, S, D_ple)
        gate = F.gelu(self.gate(hidden_states))
        fused = self.proj(gate * per_layer_embedding)
        fused_norm = self.norm(fused)
        return hidden_states + fused_norm

# --- END: New Architectural Modules for Gemini-MLA ---


class GeminiMlaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len_cached = 0
        
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class GeminiMlaLinearScalingRotaryEmbedding(GeminiMlaRotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class GeminiMlaDynamicNTKScalingRotaryEmbedding(GeminiMlaRotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class GeminiMlaYarnRotaryEmbedding(GeminiMlaRotaryEmbedding):

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GeminiMlaMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj, None


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts))
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.scoring_func == "softmax":
            scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Unsupported scoring function for MoE gating: {self.scoring_func}")

        if self.topk_method == "noaux_tc" and not self.training:
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
            )
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
            _, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
            topk_weight = scores.gather(1, topk_idx)
        else:
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor
        
        aux_loss = None
        if self.training:
            router_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
            T, E = router_probs.shape
            one_hot_mask = F.one_hot(topk_idx, num_classes=E).sum(dim=1).float()
            tokens_per_expert_fraction = one_hot_mask.mean(dim=0)
            prob_mass_per_expert_fraction = router_probs.mean(dim=0)
            aux_loss = (tokens_per_expert_fraction * prob_mass_per_expert_fraction).sum() * self.n_routed_experts

        return topk_idx, topk_weight, aux_loss


class GeminiMlaMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        if hasattr(config, "ep_size") and config.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList(
                [
                    (
                        GeminiMlaMLP(
                            config, intermediate_size=config.moe_intermediate_size
                        )
                        if i >= self.ep_rank * self.experts_per_rank
                        and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.ModuleList(
                [
                    GeminiMlaMLP(
                        config, intermediate_size=config.moe_intermediate_size
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = GeminiMlaMLP(
                config=config, intermediate_size=intermediate_size
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if not self.training:
            y = self.moe_infer(flat_hidden_states, topk_idx, topk_weight).view(*orig_shape)
        else:
            y = torch.zeros_like(flat_hidden_states)
            dispatched_expert_idx = topk_idx.flatten()
            dispatched_weights = topk_weight.flatten()
            token_indices = torch.arange(flat_hidden_states.shape[0], device=flat_hidden_states.device).repeat_interleave(self.num_experts_per_tok)

            for i in range(self.config.n_routed_experts):
                expert_mask = (dispatched_expert_idx == i)
                if not expert_mask.any():
                    continue

                tokens_to_expert = token_indices[expert_mask]
                weights_for_expert = dispatched_weights[expert_mask]
                expert_input = flat_hidden_states[tokens_to_expert]
                
                expert_output, _ = self.experts[i](expert_input)
                
                weighted_output = expert_output * weights_for_expert.unsqueeze(-1)
                y.index_add_(0, tokens_to_expert, weighted_output.to(y.dtype))

            y = y.view(*orig_shape)

        if self.shared_experts is not None:
            shared_output, _ = self.shared_experts(identity)
            y = y + shared_output
            
        return y, aux_loss

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        if self.ep_size > 1:
            logger.warning_once("Distributed moe_infer is not fully implemented in this training script. Falling back to a non-distributed version.")
        
        final_output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            if expert is None: continue
            
            token_mask = (topk_ids == i).any(dim=-1)
            if not token_mask.any():
                continue
            
            expert_weights = (topk_ids == i).float() * topk_weight
            summed_weights = expert_weights.sum(dim=-1, keepdim=True)
            expert_input = x[token_mask]
            expert_output, _ = expert(expert_input)
            weighted_output = expert_output * summed_weights[token_mask]
            final_output.index_add_(0, torch.where(token_mask)[0], weighted_output)
            
        return final_output


class GeminiMlaAttention(nn.Module):
    def __init__(self, config: GeminiMlaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        if self.q_lora_rank is None or self.q_lora_rank == 0:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = GeminiMlaRMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = GeminiMlaRMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None and "mscale_all_dim" in self.config.rope_scaling:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = GeminiMlaRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = GeminiMlaLinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = GeminiMlaDynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = GeminiMlaYarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None or self.q_lora_rank == 0:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class GeminiMlaFlashAttention2(GeminiMlaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None or self.q_lora_rank == 0:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = (
                    self.q_proj.weight.dtype
                    if self.q_lora_rank is None or self.q_lora_rank == 0
                    else self.q_a_proj.weight.dtype
                )

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, we will cast back the input in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.softmax_scale,
        )
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.v_head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        causal = self.is_causal and (query_length != 1 or not self._flash_attn_uses_top_left_mask)

        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, -1, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, -1, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


ATTENTION_CLASSES = {
    "eager": GeminiMlaAttention,
    "flash_attention_2": GeminiMlaFlashAttention2,
}


class GeminiMlaDecoderLayer(nn.Module):
    def __init__(self, config: GeminiMlaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = (
            GeminiMlaMoE(config)
            if (
                config.n_routed_experts is not None
                and config.n_routed_experts > 0
                and layer_idx is not None
                and layer_idx >= config.first_k_dense_replace
                and (layer_idx - config.first_k_dense_replace) % config.moe_layer_freq == 0
            )
            else GeminiMlaMLP(config)
        )
        self.input_layernorm = GeminiMlaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GeminiMlaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # --- Gemini-MLA Feature Modules ---
        if config.use_laurel_residual:
            self.laurel_block_attn = GeminiMlaLaurelBlock(config)
            self.laurel_block_mlp = GeminiMlaLaurelBlock(config)
        else:
            self.laurel_block_attn = None
            self.laurel_block_mlp = None

        if config.use_per_layer_embeddings:
            self.per_layer_fusion = GeminiMlaPerLayerFusion(config)
        else:
            self.per_layer_fusion = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        per_layer_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        
        # Residual connection for Attention
        if self.laurel_block_attn is not None:
            hidden_states = self.laurel_block_attn(residual) + attn_output
        else:
            hidden_states = residual + attn_output

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output, aux_loss = self.mlp(hidden_states)
        
        # Residual connection for MLP
        if self.laurel_block_mlp is not None:
            hidden_states = self.laurel_block_mlp(residual) + mlp_output
        else:
            hidden_states = residual + mlp_output

        # Per-layer Embedding Fusion
        if self.per_layer_fusion is not None and per_layer_embeddings is not None:
            # Select the embedding for the current layer
            # per_layer_embeddings shape: (B, S, num_layers, D_ple)
            current_layer_embedding = per_layer_embeddings[:, :, self.layer_idx, :]
            hidden_states = self.per_layer_fusion(hidden_states, current_layer_embedding)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        
        outputs += (aux_loss,)

        return outputs


GeminiMla_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    This model is a custom architecture, Gemini-MLA, which integrates several novel features:
    1. Scaled Input Embeddings: The initial token embeddings can be scaled by a fixed factor.
    2. Enhanced Residual Connections (Laurel): The residual path in each transformer layer is enhanced with a low-rank transformation.
    3. Per-Layer Embeddings: Unique embeddings are generated for each token at each layer and fused into the hidden state.
    Parameters:
        config ([`GeminiMlaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare GeminiMla Model outputting raw hidden-states without any specific head on top.",
    GeminiMla_START_DOCSTRING,
)
class GeminiMlaPreTrainedModel(PreTrainedModel):
    config_class = GeminiMlaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GeminiMlaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


GeminiMla_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up
            sequential decoding.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare GeminiMla Model outputting raw hidden-states without any specific head on top.",
    GeminiMla_START_DOCSTRING,
)
class GeminiMlaModel(GeminiMlaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GeminiMlaDecoderLayer`]
    Args:
        config: GeminiMlaConfig
    """

    def __init__(self, config: GeminiMlaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # --- Gemini-MLA Feature: Scaled Input Embeddings ---
        if config.use_scaled_embeddings:
            self.embed_tokens = GeminiMlaScaledEmbedder(
                config.vocab_size, config.hidden_size, config.embedding_scale_factor, self.padding_idx
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size, self.padding_idx
            )
        
        # --- Gemini-MLA Feature: Per-Layer Embeddings ---
        if config.use_per_layer_embeddings:
            self.per_layer_embedder = GeminiMlaPerLayerEmbedder(config)
        else:
            self.per_layer_embedder = None

        self.layers = nn.ModuleList(
            [
                GeminiMlaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = GeminiMlaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(GeminiMla_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndAuxLoss]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # --- Gemini-MLA Feature: Per-Layer Embeddings Calculation ---
        per_layer_embeddings = None
        if self.per_layer_embedder is not None:
            if input_ids is None:
                raise ValueError("Per-layer embeddings require `input_ids` and cannot be used with `inputs_embeds`.")
            # Handle padding tokens if necessary before passing to embedding layers
            ple_input_ids = input_ids.clone()
            ple_input_ids[ple_input_ids == -100] = self.config.pad_token_id if self.config.pad_token_id is not None else 0
            per_layer_embeddings = self.per_layer_embedder(ple_input_ids)

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            attention_mask_2d = attention_mask
        else:
            attention_mask_2d = None
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=getattr(self.config, "sliding_window", None),
            )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_aux_losses = [] if self.training else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpoint(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask if not self._use_flash_attention_2 else attention_mask_2d,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    per_layer_embeddings, # Pass to layer
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask if not self._use_flash_attention_2 else attention_mask_2d,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    per_layer_embeddings=per_layer_embeddings, # Pass to layer
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if self.training:
                aux_loss = layer_outputs[-1]
                if aux_loss is not None:
                    all_aux_losses.append(aux_loss)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
        if not return_dict:
            outputs = [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if all_aux_losses is not None:
                outputs.append(all_aux_losses)
            return tuple(v for v in outputs if v is not None)
            
        return BaseModelOutputWithPastAndAuxLoss(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            aux_losses=all_aux_losses,
        )

# --- START: MTP modules ---

class GeminiMlaPredictorLayer(nn.Module):
    def __init__(self, config: GeminiMlaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.enorm = GeminiMlaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = GeminiMlaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        
        self.mtp_block = GeminiMlaDecoderLayer(config, layer_idx=None)

    def forward(
        self,
        previous_spec_embeds: torch.Tensor,
        backbone_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        previous_spec_embeds = self.enorm(previous_spec_embeds)
        backbone_hidden_states = self.hnorm(backbone_hidden_states)

        fused_hidden_states = self.eh_proj(
            torch.cat([previous_spec_embeds, backbone_hidden_states], dim=-1)
        )

        layer_outputs = self.mtp_block(
            hidden_states=fused_hidden_states,
            attention_mask=None,
            position_ids=None,
            use_cache=False,
            output_attentions=False,
            # MTP layers do not use per-layer embeddings from the input sequence
            per_layer_embeddings=None,
        )
        
        return layer_outputs[0]


class GeminiMlaPredictor(nn.Module):
    def __init__(self, config: GeminiMlaConfig):
        super().__init__()
        self.config = config
        self.num_predict_layers = config.num_nextn_predict_layers
        
        self.layers = nn.ModuleList(
            [GeminiMlaPredictorLayer(config) for _ in range(self.num_predict_layers)]
        )

    def forward(
        self,
        backbone_hidden_states: torch.Tensor,
        embed_tokens: nn.Module,
        lm_head: nn.Linear,
        labels: torch.LongTensor,
    ) -> torch.Tensor:
        B, S_prime, H = backbone_hidden_states.shape
        device = backbone_hidden_states.device
        all_mtp_logits = []

        previous_spec_embeds = torch.zeros_like(backbone_hidden_states)

        for i in range(self.num_predict_layers):
            current_mtp_hs = self.layers[i](previous_spec_embeds, backbone_hidden_states)
            current_logits = lm_head(current_mtp_hs)
            all_mtp_logits.append(current_logits)

            if i < self.num_predict_layers - 1:
                start_idx = 1 + i
                end_idx = S_prime + 1 + i
                teacher_force_labels = labels[:, start_idx:end_idx].contiguous()
                
                teacher_force_labels = teacher_force_labels.to(device)
                teacher_force_labels = torch.clamp(teacher_force_labels, min=0)

                previous_spec_embeds = embed_tokens(teacher_force_labels)

        return torch.stack(all_mtp_logits, dim=0).permute(1, 2, 0, 3)

# --- END: MTP modules ---

class GeminiMlaForCausalLM(GeminiMlaPreTrainedModel,GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GeminiMlaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.num_nextn_predict_layers > 0:
            self.mtp_predictor = GeminiMlaPredictor(config)
        else:
            self.mtp_predictor = None

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(GeminiMla_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPastAndAuxLoss]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, config.vocab_size]` 
                or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is 
                only computed for the tokens with labels in `[0, config.vocab_size]`.
        Returns:
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)

            total_loss = lm_loss

            if self.mtp_predictor is not None and self.config.mtp_beta > 0 and self.training:
                N = self.config.num_nextn_predict_layers
                if labels.shape[1] > 1 + N:
                    mtp_hidden_states = hidden_states[:, :-(1 + N), :].contiguous()
                    S_prime = mtp_hidden_states.shape[1]
                    
                    if S_prime > 0:
                        mtp_logits = self.mtp_predictor(
                            backbone_hidden_states=mtp_hidden_states,
                            embed_tokens=self.model.embed_tokens,
                            lm_head=self.lm_head,
                            labels=labels,
                        )

                        mtp_labels_list = []
                        for i in range(N):
                            start_idx = 2 + i
                            end_idx = S_prime + 2 + i
                            mtp_labels_list.append(labels[:, start_idx:end_idx])
                        
                        mtp_labels = torch.stack(mtp_labels_list, dim=2)

                        mtp_logits_for_loss = mtp_logits.view(-1, self.config.vocab_size)
                        mtp_labels_for_loss = mtp_labels.contiguous().view(-1).to(mtp_logits_for_loss.device)
                        
                        mtp_loss = loss_fct(mtp_logits_for_loss, mtp_labels_for_loss)
                        total_loss += self.config.mtp_beta * mtp_loss

            aux_losses = outputs[-1] if not return_dict else outputs.aux_losses
            if aux_losses is not None and len(aux_losses) > 0 and self.training:
                aux_loss = sum(aux_losses)
                total_loss += self.config.aux_loss_alpha * aux_loss
            
            loss = total_loss


        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndAuxLoss(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            aux_losses=outputs.aux_losses if return_dict else outputs[-1],
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length() if hasattr(past_key_values, 'get_max_length') else None
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


@add_start_docstrings(
    """
    The GeminiMla Model transformer with a sequence classification head on top (linear layer).
    """,
    GeminiMla_START_DOCSTRING,
)
class GeminiMlaForSequenceClassification(GeminiMlaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = GeminiMlaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(GeminiMla_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0] if not return_dict else transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + (transformer_outputs[1:] if not return_dict else (
                transformer_outputs.past_key_values,
                transformer_outputs.hidden_states,
                transformer_outputs.attentions,
            ))
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
