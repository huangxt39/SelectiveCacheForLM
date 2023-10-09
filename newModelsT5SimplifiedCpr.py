import copy
import math
import os
import warnings
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from matplotlib import pyplot as plt
import shutil
import random
import re

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
    ModelOutput,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5LayerFF

# logger = logging.get_logger(__name__)


@dataclass
class DecoderModelOutput(ModelOutput):

    loss: torch.FloatTensor = None
    logits: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    token_nums:  Optional[torch.LongTensor] = None

@dataclass
class DecoderStackOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    present_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    recon_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    all_select_mask: Optional[Tuple[torch.FloatTensor]] = None


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


try:
    from apex.normalization import FusedRMSNorm

    T5LayerNorm = FusedRMSNorm  # noqa

    print("Discovered apex.normalization.FusedRMSNorm - will use it instead of T5LayerNorm")
except ImportError:
    # using the normal T5LayerNorm
    pass
except Exception:
    print("discovered apex but it failed to load, falling back to T5LayerNorm")
    pass

ALL_LAYERNORM_LAYERS.append(T5LayerNorm)

FLAG_TEST = True
TOKENIZER = None
MASKED_VOCAB = None

class T5Attention(nn.Module):
    def __init__(self, config: T5Config, block_size, xl_cache_size, sec_cache_size, compress_rate, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.compress_rate = compress_rate

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)

        self.gradient_checkpointing = False

        self.xl_cache = {}  
        self.sec_cache = {} 
        self.xl_cache_size = xl_cache_size if xl_cache_size > 0 else 0
        self.sec_cache_size = sec_cache_size if sec_cache_size > 0 else 0

        self.compressor = nn.Conv1d(self.d_model, self.d_model, kernel_size=compress_rate, stride=compress_rate)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):

        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets = torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, max_dist_len, device):
        """Compute binned relative position bias"""
        query_position = torch.arange(key_length-query_length, key_length, dtype=torch.long, device=device)[:, None]
        key_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = key_position - query_position  # shape (query_length, key_length)
        
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        if max_dist_len > 0:
            relative_position_bucket[:, :max_dist_len] = self.relative_attention_num_buckets - 1
            
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def make_causal_mask(self, query_length, key_length, device, dtype):
        # causal_mask = torch.tril(torch.ones((key_length, key_length), device=device, dtype=dtype))[key_length-query_length:key_length]
        causal_mask = torch.cat([torch.ones((query_length, key_length-query_length), device=device, dtype=dtype), \
                                 torch.tril(torch.ones((query_length, query_length), device=device, dtype=dtype))], dim=1 )
        causal_mask = (1.0 - causal_mask) * torch.finfo(dtype).min
        return causal_mask.unsqueeze(0).unsqueeze(0)
    
    def make_doc_mask(self, query_doc_ids, key_doc_ids, dtype):
        # batch_size q_len
        # batch_size k_len
        doc_mask = (query_doc_ids.unsqueeze(-1) == key_doc_ids.unsqueeze(2)).float()
        # self.examine_weights(doc_mask.unsqueeze(1), 'doc_m')
        doc_mask = (1.0 - doc_mask) * torch.finfo(dtype).min
        return doc_mask
    
    def update_sec_cache(self, query_states, value_states, doc_ids, hidden_states, raw_scores, device_name):
       
        # get inital cache
        if device_name in self.sec_cache:
            old_k, old_v, old_d = self.sec_cache[device_name]
        else:
            shape = query_states.size()[:2] + (self.sec_cache_size,) + value_states.size()[3:4]
            old_k = torch.zeros(shape, dtype=value_states.dtype, device=value_states.device)
            old_v = torch.zeros_like(old_k)
            old_d = torch.full(shape[:-1], -1, dtype=doc_ids.dtype, device=doc_ids.device)

        def no_grad_to_linear(module, states):
            batch_size = states.size(0)
            states = torch.matmul(states.view(-1, states.size(-1)), module.weight.detach().clone().transpose(0,1))
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        # original 
        with torch.no_grad():
            orig_attn = nn.functional.softmax(raw_scores.float(), dim=-1).type_as(
                raw_scores
            )  # (batch_size, n_heads, seq_length, key_length)
    
            orig_output = torch.matmul(orig_attn, value_states)  # (batch_size, seq_length, dim)

            # orig_k = no_grad_to_linear(self.k, hidden_states)
            # orig_v = no_grad_to_linear(self.v, hidden_states)
            
            # orig_scores = torch.matmul(query_states, orig_k.transpose(3, 2))
            # orig_attn = nn.functional.softmax(orig_scores.float(), dim=-1).type_as(orig_scores)  
        
            # orig_output = torch.matmul(orig_attn, orig_v) 
        
        compressed_h = self.compressor(hidden_states.transpose(1,2)).transpose(1,2).contiguous()
        
        compressed_k = no_grad_to_linear(self.k, compressed_h)
        compressed_v = no_grad_to_linear(self.v, compressed_h)
        
        compressed_scores = torch.matmul(query_states, compressed_k.transpose(3, 2))
        compressed_attn = nn.functional.softmax(compressed_scores.float(), dim=-1).type_as(compressed_scores)  
    
        compressed_output = torch.matmul(compressed_attn, compressed_v) 
        
        recon_loss = nn.functional.mse_loss(orig_output, compressed_output)

        # update cache
        add_len = compressed_k.size(2)
        assert add_len <= self.sec_cache_size
        compressed_d = doc_ids[:,:, :add_len*self.compress_rate].contiguous().view(doc_ids.size(0), doc_ids.size(1), -1, self.compress_rate)[:,:,:,0]
        old_k = torch.cat([old_k[:, :, add_len:], compressed_k.detach()], dim=2)
        old_v = torch.cat([old_v[:, :, add_len:], compressed_v.detach()], dim=2)
        old_d = torch.cat([old_d[:, :, add_len:], compressed_d], dim=2)
        self.sec_cache[device_name] = (old_k, old_v, old_d)

        return recon_loss
        
    
    def forward(
        self,
        hidden_states,
        position_bias=None,
        doc_ids=None,
        output_attentions=False,
    ):
        # Input is (batch_size, seq_length, dim)
        batch_size, seq_length = hidden_states.shape[:2]
        doc_ids = doc_ids.unsqueeze(1).expand(-1, self.n_heads, -1)

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))

        current_doc_ids = doc_ids

        device_name = str(hidden_states.device)
        if self.xl_cache_size > 0 and device_name in self.xl_cache:
            cache_key, cache_value, cache_doc_ids, cache_hidden = self.xl_cache[device_name]
            # print("xl_cache", device_name, cache_key.size())
            key_states = torch.cat([cache_key, key_states], dim=2)
            value_states = torch.cat([cache_value, value_states], dim=2)
            doc_ids = torch.cat([cache_doc_ids, doc_ids], dim=2)
            hidden_states = torch.cat([cache_hidden, hidden_states], dim=1)
            # print("concated: ", device_name, key_states.size())

        if self.sec_cache_size > 0 and device_name in self.sec_cache:
            cache_key, cache_value, cache_doc_ids = self.sec_cache[device_name]
            # print("sec_cache", device_name, cache_key.size())
            key_states = torch.cat([cache_key, key_states], dim=2)
            value_states = torch.cat([cache_value, value_states], dim=2)
            doc_ids = torch.cat([cache_doc_ids, doc_ids], dim=2)


        # self.examine_ids(current_input_ids, input_ids)

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        # self.examine_weights(scores, "scores")

        sec_key_num = 0
        if self.sec_cache_size > 0 and device_name in self.sec_cache:
            sec_key_num += self.sec_cache_size

        to_compress_end = -self.xl_cache_size if self.xl_cache_size > 0 else scores.size(-1)
        raw_scores = scores[:,:,:,sec_key_num:to_compress_end].contiguous().detach()
        
        if position_bias is None:
            assert self.has_relative_attention_bias
            
            position_bias = self.compute_bias(query_states.size(2), key_states.size(2), sec_key_num, device=scores.device)
            # (1, n_heads, query_length, key_length)
            # self.examine_weights(position_bias, 'pos_bias')
            
            mask = self.make_causal_mask(query_states.size(2), key_states.size(2), device=scores.device, dtype=scores.dtype)    # (1, 1, query_length, key_length)
            # self.examine_weights(mask/abs(torch.finfo(scores.dtype).min), 'causal_m')
            position_bias = position_bias + mask    
        scores += position_bias

        doc_mask = self.make_doc_mask(current_doc_ids, doc_ids, scores.dtype)
        scores += doc_mask
        

        # self.examine_weights(scores/abs(torch.finfo(scores.dtype).min), "scores_final")
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
 
        # self.examine_weights(attn_weights, 'attn_w')

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, position_bias )

        if self.sec_cache_size > 0:
            recon_loss = self.update_sec_cache(query_states.detach(), 
                                  value_states[:,:,sec_key_num:to_compress_end].contiguous().detach(),
                                  doc_ids[:,:,sec_key_num:to_compress_end].contiguous(),
                                  hidden_states[:,:to_compress_end].contiguous().detach(),
                                  raw_scores, device_name)
        if self.xl_cache_size > 0:
            self.xl_cache[device_name] = ( key_states[:, :, -self.xl_cache_size:].contiguous().detach(), \
                                        value_states[:, :, -self.xl_cache_size:].contiguous().detach(), \
                                        doc_ids[:, :, -self.xl_cache_size:].contiguous(), \
                                        hidden_states[:, -self.xl_cache_size:].contiguous().detach() )
            
        
        outputs = outputs + (recon_loss,)
        
        if output_attentions:
            outputs = outputs + (attn_weights,)
        
        return outputs

    def examine_weights(self, weights, desc):
        if FLAG_TEST:
            device_name = str(weights.device)

            BATCH_IDX = 0
            HEAD_IDX = 0
            
            
            attn_map = weights[BATCH_IDX, HEAD_IDX].detach().cpu() * 3

            fig, ax = plt.subplots()
            ax.imshow(attn_map.numpy())

            ax.set_title("attention map")
            fig.tight_layout()
            counter = 1
            while os.path.exists(f'./images/{device_name}-{desc}-Iter{counter}.png'):
                counter += 1

            fig.savefig(f'./images/{device_name}-{desc}-Iter{counter}.png')

    def examine_ids(self, current_input_ids, input_ids):
        if FLAG_TEST:
            device_name = str(current_input_ids.device)
            counter = 1
            while os.path.exists(f'./images/{device_name}-Iter{counter}.txt'):
                counter += 1

            with open(f'./images/{device_name}-Iter{counter}.txt', 'w') as f:
                for i in range(current_input_ids.size(0)):
                    for j in range(current_input_ids.size(1)):
                        f.write(str(current_input_ids[i,j].tolist()))
                        f.write('\n')
                        f.write(str(input_ids[i,j].tolist()))
                        f.write('\n' + '='*100 + '\n')


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, block_size, xl_cache_size, sec_cache_size, compress_rate, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, block_size, xl_cache_size, sec_cache_size, compress_rate, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        position_bias=None,
        doc_ids=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            position_bias=position_bias,
            doc_ids=doc_ids,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

class PlaceHolder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        pass

class T5Block(nn.Module):
    def __init__(self, config, block_size, xl_cache_size, sec_cache_size, compress_rate, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        assert config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, block_size, xl_cache_size, sec_cache_size, compress_rate, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(PlaceHolder())
            
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        position_bias=None,
        doc_ids=None,
        output_attentions=False,
    ):

        self_attention_outputs = self.layer[0](
            hidden_states,
            position_bias=position_bias,
            doc_ids=doc_ids,
            output_attentions=output_attentions,
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[1:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,) + attention_outputs

        return outputs  # hidden-states, position bias,  kv (self-attention weights)



class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens, block_size, xl_cache_size, sec_cache_size, compress_rate):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, block_size, xl_cache_size, sec_cache_size, compress_rate, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        doc_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    
        assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
        inputs_embeds = self.embed_tokens(input_ids)

        all_recon_loss = 0 
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    position_bias=position_bias,
                    doc_ids=doc_ids,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # (attn_output, position_bias, recon_loss, output_attentions
            hidden_states = layer_outputs[0]
            # We share the position biases between the layers - the first layer store them
            position_bias = layer_outputs[1]

            all_recon_loss += layer_outputs[2]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[-1],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_recon_loss,
                    all_hidden_states,
                    all_attentions,
                ]
                if v is not None
            )
        return DecoderStackOutput(
            last_hidden_state=hidden_states,
            recon_loss=all_recon_loss,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

class T5ModelDecoderCacheCompressor(T5PreTrainedModel):

    def __init__(self, config: T5Config, block_size, xl_cache_size, sec_cache_size, compress_rate, tokenizer=None):
        super().__init__(config)

        print("params:\t", "xl_cache", xl_cache_size, "sec_cache", sec_cache_size, "compress", compress_rate)

        self.block_size = block_size
        self.xl_cache_size = xl_cache_size
        self.sec_cache_size = sec_cache_size

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared, block_size, xl_cache_size, sec_cache_size, compress_rate)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None                

        if FLAG_TEST:
            for i in range(self.config.num_layers):
                self.decoder.block[i].layer[0].SelfAttention.current_layer = i


    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.decoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.decoder.block))
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.decoder.deparallelize()
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.decoder
    
    def reset_cache(self, batch_size_per_device=None):
        print(f'empty cache...')
        for layer_module in self.decoder.block:
            layer_module.layer[0].SelfAttention.xl_cache = {}
            layer_module.layer[0].SelfAttention.sec_cache = {}


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        fix_duration: Optional[torch.LongTensor] = None,
        doc_ids:  Optional[torch.LongTensor] = None,
        labels: torch.LongTensor = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_select_mask: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            doc_ids=doc_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs.last_hidden_state
        loss = decoder_outputs.recon_loss * 1e-3
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            self.lm_head = self.lm_head.to(self.decoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.config.d_model**-0.5)

        if labels is not None:
            lm_logits = self.lm_head(sequence_output)

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(ignore_index=-100)

            loss = loss + loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
            token_nums = (shift_labels != -100).sum()
            if token_nums.item() == 0:
                loss = torch.zeros_like(loss)
        else:
            lm_logits = None
            token_nums = None

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return DecoderModelOutput(
            loss=loss,
            logits=lm_logits,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            token_nums=token_nums,
        )
