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
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.distributions as tdist

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
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


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
    def __init__(self, config: T5Config, block_size, xl_cache_size, sec_cache_size, has_relative_attention_bias=False):
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
        assert sec_cache_size == block_size

        if self.sec_cache_size > 0:
            self.sec_cache_bias = nn.Parameter(torch.zeros(1, self.n_heads, 1, 1))

        input_dim = self.d_model
        self.replacer = nn.Sequential(nn.Linear(input_dim + 8, input_dim//4), nn.ReLU(), nn.Linear(input_dim//4, 1, bias=False))
        
        self.register_buffer('t_emb', 2 ** torch.arange(-7, 1, dtype=torch.float))
        self.HC_beta = 2/3
        self.HC_gamma = -0.1
        self.HC_zeta = 1.1
        

    def init_sec_cache(self, batch_size_per_device):
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            shape = (batch_size_per_device, self.n_heads, self.sec_cache_size, self.key_value_proj_dim)

            k = torch.zeros(shape, device=device)
            v = torch.zeros(shape, device=device)
            h = torch.zeros(batch_size_per_device, self.sec_cache_size, self.d_model, device=device)
            d = torch.full(shape[:-1], -1, device=device)
            
            id = torch.full(shape[:-1], 0, device=device)
            t = torch.full((batch_size_per_device, self.sec_cache_size), 0, device=device)
            self.sec_cache[f'cuda:{i}'] = (k, v, h, d, id, t)      

    def sample_z_mask(self, log_alpha):
        u = torch.rand_like(log_alpha).clamp(min=1e-8)  # avoid 0
        s = torch.sigmoid((torch.log(u) - torch.log(1-u) + log_alpha) / self.HC_beta) * (self.HC_zeta - self.HC_gamma) + self.HC_gamma
        z = s.clamp(min=1e-8, max=1-1e-8)
        return z

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
    
    def update_cache(self, key_states, value_states, hidden_states, doc_ids, device_name, input_ids, timestamp):
       
        if self.xl_cache_size > 0 and device_name in self.xl_cache:
            to_sec_cache = [ key_states[:, :, :-self.xl_cache_size].detach(), \
                                    value_states[:, :, :-self.xl_cache_size].detach(), \
                                    hidden_states[:, :-self.xl_cache_size].detach(), \
                                    doc_ids[:, :, :-self.xl_cache_size],\
                                    input_ids[:, :, :-self.xl_cache_size] ]
            
            to_sec_cache[0] = torch.cat([self.xl_cache[device_name][0], to_sec_cache[0]], dim=2)
            to_sec_cache[1] = torch.cat([self.xl_cache[device_name][1], to_sec_cache[1]], dim=2)
            to_sec_cache[2] = torch.cat([self.xl_cache[device_name][2], to_sec_cache[2]], dim=1)
            to_sec_cache[3] = torch.cat([self.xl_cache[device_name][3], to_sec_cache[3]], dim=2)
            
            to_sec_cache[4] = torch.cat([self.xl_cache[device_name][4], to_sec_cache[4]], dim=2)
        else:
            to_sec_cache = [ key_states.detach(), \
                            value_states.detach(), \
                            hidden_states.detach(), \
                            doc_ids, input_ids ]
    
        old_k, old_v, old_h, old_d, old_id, old_t = self.sec_cache[device_name]

        if to_sec_cache[2].size(1) != old_h.size(1):
            # print("end of epoch, return")
            return      

        with torch.no_grad():
            old_t_diff = (timestamp + 1 - old_t.unsqueeze(-1)) * self.t_emb.view(1, 1, -1)
            to_cache_diff = torch.ones_like(old_t).unsqueeze(-1) * self.t_emb.view(1, 1, -1)
 
            log_alpha = self.replacer(torch.cat([to_sec_cache[2], to_cache_diff], dim=-1)) - self.replacer(torch.cat([old_h, old_t_diff], dim=-1))

            indicator = (log_alpha > 0).unsqueeze(1) | (to_sec_cache[3] != old_d).unsqueeze(-1)


            if FLAG_TEST and random.randint(0, 10000) == 0:
                hist, bins = torch.histogram(log_alpha.float().cpu(), bins=5)
                print(hist, bins)
                print('layer:', self.current_layer, 'mean log alpha:', log_alpha.mean().item(), 'std log alpha:', log_alpha.std().item(), 'select ratio:', ((log_alpha>0).sum() / log_alpha.numel()).item())
                BATCH_IDX = 0
                PRINT_LEN = 100
                print(' '.join([ f'|<{token1}> {token2}' if chosen>0 else f'|{token1} <{token2}>' for chosen, token1, token2 in zip(log_alpha[BATCH_IDX, :PRINT_LEN, 0].tolist(),\
                                                                                                            TOKENIZER.convert_ids_to_tokens(to_sec_cache[4][BATCH_IDX, 0, :PRINT_LEN].tolist()), \
                                                                                                            TOKENIZER.convert_ids_to_tokens(old_id[BATCH_IDX, 0, :PRINT_LEN].tolist()))]))
        
            new_k = torch.where(indicator, to_sec_cache[0], old_k.type_as(to_sec_cache[0])) 
            new_v = torch.where(indicator, to_sec_cache[1], old_v.type_as(to_sec_cache[1]))
            new_h = torch.where(indicator[:,0], to_sec_cache[2], old_h.type_as(to_sec_cache[2]))
            new_d = torch.where(indicator.squeeze(-1), to_sec_cache[3], old_d.type_as(to_sec_cache[3]))
            new_id = torch.where(indicator.squeeze(-1), to_sec_cache[4], old_id.type_as(to_sec_cache[4]))
            new_t = torch.where(indicator[:, 0, :, 0], timestamp, old_t)
        
        self.sec_cache[device_name] = (new_k, new_v, new_h, new_d, new_id, new_t)

        if self.xl_cache_size > 0:
            self.xl_cache[device_name] = ( key_states[:, :, -self.xl_cache_size:].contiguous().detach(), \
                                        value_states[:, :, -self.xl_cache_size:].contiguous().detach(), \
                                        hidden_states[:, -self.xl_cache_size:].contiguous().detach(), \
                                        doc_ids[:, :, -self.xl_cache_size:].contiguous(),\
                                        input_ids[:, :, -self.xl_cache_size:].contiguous() )
            
    
    def forward(
        self,
        hidden_states,
        position_bias=None,
        doc_ids=None,
        past_kv=None,
        timestamp=None,
        output_attentions=False,
        input_ids=None,
    ):
        # Input is (batch_size, seq_length, dim)
        batch_size, seq_length = hidden_states.shape[:2]
        doc_ids = doc_ids.unsqueeze(1).expand(-1, self.n_heads, -1)
        input_ids = input_ids.unsqueeze(1).expand(-1, self.n_heads, -1)

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

        current_key_states = key_states
        current_value_states = value_states
        current_doc_ids = doc_ids
        
        current_input_ids = input_ids
        current_timestamp = timestamp

        device_name = str(hidden_states.device)
        if self.xl_cache_size > 0 and device_name in self.xl_cache:
            cache_key, cache_value, _, cache_doc_ids, cache_input_ids = self.xl_cache[device_name]
            # print("xl_cache", device_name, cache_key.size())
            key_states = torch.cat([cache_key, key_states], dim=2)
            value_states = torch.cat([cache_value, value_states], dim=2)
            doc_ids = torch.cat([cache_doc_ids, doc_ids], dim=2)
            input_ids = torch.cat([cache_input_ids, input_ids], dim=2)
            # print('aaa', input_ids.size())

        if self.sec_cache_size > 0 and device_name in self.sec_cache:
            cache_key, cache_value, cache_hidden, cache_doc_ids, cache_input_ids, cache_timestamp = self.sec_cache[device_name]
            # print("sec_cache", device_name, cache_key.size())
            if past_kv is not None:
                # rearrange
                s = torch.randint(0, cache_hidden.size(1), ()).item()
                exceed_len = s + past_kv[2].size(1) - cache_hidden.size(1)
                if exceed_len >= 0:
                    cache_hidden = torch.cat([cache_hidden[:, s:], cache_hidden[:, :exceed_len]], dim=1)
                    # selected_doc_ids = torch.cat([cache_doc_ids[:, :, s:], cache_doc_ids[:, :, :exceed_len]], dim=2)
                    cache_timestamp = torch.cat([cache_timestamp[:, s:], cache_timestamp[:, :exceed_len]], dim=1)
                    # selected_input_ids = torch.cat([cache_input_ids[:, :, s:], cache_input_ids[:, :, :exceed_len]], dim=2)

                    cache_key = torch.cat([cache_key[:, :, s:], cache_key[:, :, :exceed_len], cache_key[:, :, exceed_len:s]], dim=2)
                    cache_value = torch.cat([cache_value[:, :, s:], cache_value[:, :, :exceed_len], cache_value[:, :, exceed_len:s]], dim=2)
                    cache_doc_ids = torch.cat([cache_doc_ids[:, :, s:], cache_doc_ids[:, :, :exceed_len], cache_doc_ids[:, :, exceed_len:s]], dim=2)
                    cache_input_ids = torch.cat([cache_input_ids[:, :, s:], cache_input_ids[:, :, :exceed_len], cache_input_ids[:, :, exceed_len:s]], dim=2)
                    
                    selected_doc_ids = cache_doc_ids[:, :, :cache_hidden.size(1)]
                else:
                    cache_hidden = cache_hidden[:, s:exceed_len].contiguous()
                    # selected_doc_ids = cache_doc_ids[:, :, s:exceed_len].contiguous()
                    cache_timestamp = cache_timestamp[:, s:exceed_len].contiguous()
                    # selected_input_ids = cache_input_ids[:, :, s:exceed_len].contiguous()

                    cache_key = torch.cat([cache_key[:, :, s:exceed_len], cache_key[:, :, :s], cache_key[:, :, exceed_len:]], dim=2)
                    cache_value = torch.cat([cache_value[:, :, s:exceed_len], cache_value[:, :, :s], cache_value[:, :, exceed_len:]], dim=2)
                    cache_doc_ids = torch.cat([cache_doc_ids[:, :, s:exceed_len], cache_doc_ids[:, :, :s], cache_doc_ids[:, :, exceed_len:]], dim=2)
                    cache_input_ids = torch.cat([cache_input_ids[:, :, s:exceed_len], cache_input_ids[:, :, :s], cache_input_ids[:, :, exceed_len:]], dim=2)
                    
                    selected_doc_ids = cache_doc_ids[:, :, :cache_hidden.size(1)]

            key_states = torch.cat([cache_key, key_states], dim=2)
            value_states = torch.cat([cache_value, value_states], dim=2)
            doc_ids = torch.cat([cache_doc_ids, doc_ids], dim=2)
            
            input_ids = torch.cat([cache_input_ids, input_ids], dim=2)

        if past_kv is not None:
            past_k, past_v, past_h, past_d, past_id, past_t = past_kv
            past_d = past_d.unsqueeze(1).expand(-1, self.n_heads, -1)
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)
            doc_ids = torch.cat([past_d, doc_ids], dim=2)

            input_ids = torch.cat([past_id.unsqueeze(1).expand(-1, self.n_heads, -1), input_ids], dim=2)
            past_t = torch.full(past_id.size(), past_t, device=past_id.device) 


        # self.examine_ids(current_input_ids, input_ids)

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        # self.examine_weights(scores, "scores")

        sec_key_num = 0
        if self.sec_cache_size > 0 and device_name in self.sec_cache:
            sec_key_num += self.sec_cache_size
        if past_kv is not None:
            sec_key_num += past_d.size(-1)
        
        if sec_key_num > 0:
            scores += torch.cat([self.sec_cache_bias.expand(-1, -1, -1, sec_key_num).contiguous(), \
                   torch.zeros((1, self.n_heads, 1, scores.size(-1) - sec_key_num,), device=scores.device, dtype=scores.dtype)], dim=-1)

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
        
        if past_kv is not None:
              
            past_t_diff = (timestamp - past_t.unsqueeze(-1)) * self.t_emb.view(1, 1, -1)
            cache_t_diff = (timestamp - cache_timestamp.unsqueeze(-1)) * self.t_emb.view(1, 1, -1)

            log_alpha = (self.replacer(torch.cat([past_h, past_t_diff], dim=-1)) - self.replacer(torch.cat([cache_hidden, cache_t_diff], dim=-1))).squeeze(-1)

            z_value = self.sample_z_mask(log_alpha)
            z_value = torch.where((past_d != selected_doc_ids)[:, 0], 1e-8, z_value)
            shape_ = (z_value.size(0), scores.size(-1) - z_value.size(-1)*2)
            replace_mask = torch.cat([z_value.log(), (1-z_value).log(), torch.zeros(shape_, device=z_value.device)], dim=-1).unsqueeze(1).unsqueeze(1)
            # assert torch.isnan(replace_mask).sum().item() == 0
            scores += replace_mask

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

        outputs = (attn_output, position_bias, (current_key_states.detach(), current_value_states.detach(), hidden_states.detach(), timestamp))

        if output_attentions:
            outputs = outputs + (attn_weights,)

        self.update_cache(current_key_states, current_value_states, hidden_states, current_doc_ids, device_name, current_input_ids, current_timestamp)
        

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
    def __init__(self, config, block_size, xl_cache_size, sec_cache_size, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, block_size, xl_cache_size, sec_cache_size, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        position_bias=None,
        doc_ids=None,
        past_kv=None,
        timestamp=None,
        output_attentions=False,
        input_ids=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            position_bias=position_bias,
            doc_ids=doc_ids,
            past_kv=past_kv,
            timestamp=timestamp,
            output_attentions=output_attentions,
            input_ids=input_ids,
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
    def __init__(self, config, block_size, xl_cache_size, sec_cache_size, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        assert config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, block_size, xl_cache_size, sec_cache_size, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(PlaceHolder())
            
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        position_bias=None,
        doc_ids=None,
        past_kv=None,
        timestamp=None,
        output_attentions=False,
        input_ids=None,
    ):

        self_attention_outputs = self.layer[0](
            hidden_states,
            position_bias=position_bias,
            doc_ids=doc_ids,
            past_kv=past_kv,
            timestamp=timestamp,
            output_attentions=output_attentions,
            input_ids=input_ids,
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
    def __init__(self, config, embed_tokens, block_size, xl_cache_size, sec_cache_size):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, block_size, xl_cache_size, sec_cache_size, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
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
        past_key_values=None,
        timestamp=None,
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

        have_past_kv = past_key_values is not None
        past_key_values = past_key_values if have_past_kv else (None for _ in range(self.config.num_layers))
        all_present_key_values = ()
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_kv) in enumerate(zip(self.block, past_key_values)):
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
                    past_kv=past_kv,
                    timestamp=timestamp,
                    output_attentions=output_attentions,
                    input_ids=input_ids
                )

            # layer_outputs is a tuple with:
            # (attn_output, position_bias, (current_key_states, current_value_states), output_attentions, log_alpha
            hidden_states = layer_outputs[0]
            # We share the position biases between the layers - the first layer store them
            position_bias = layer_outputs[1]

            all_present_key_values = all_present_key_values + (layer_outputs[2],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)

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
                    all_present_key_values,
                    all_hidden_states,
                    all_attentions,
                ]
                if v is not None
            )
        return DecoderStackOutput(
            last_hidden_state=hidden_states,
            present_key_values=all_present_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

class T5ModelDecoderCacheReplacer(T5PreTrainedModel):

    def __init__(self, config: T5Config, block_size, xl_cache_size, sec_cache_size, snippet_size, snip_list_len, tokenizer=None):
        super().__init__(config)

        print("params:\t", "xl_cache", xl_cache_size, "sec_cache", sec_cache_size, "selector", snippet_size, snip_list_len)

        self.block_size = block_size
        self.xl_cache_size = xl_cache_size
        self.sec_cache_size = sec_cache_size

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared, block_size, xl_cache_size, sec_cache_size)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.snippet_list = {}
        self.snippet_size = snippet_size
        self.snip_list_len = snip_list_len

        self.timestamp = {}

        if FLAG_TEST and tokenizer is not None:
            global TOKENIZER
            TOKENIZER = tokenizer
                
        if FLAG_TEST and os.path.exists('./images'):
            shutil.rmtree('./images')
            os.mkdir('./images')

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
    
    def reset_cache(self, batch_size_per_device):
        print(f'empty cache...')
        for layer_module in self.decoder.block:
            layer_module.layer[0].SelfAttention.xl_cache.clear()
            layer_module.layer[0].SelfAttention.sec_cache.clear()
        self.snippet_list.clear()
        # print(torch.cuda.memory_summary())
        # torch.cuda.empty_cache()
        for layer_module in self.decoder.block:
            layer_module.layer[0].SelfAttention.init_sec_cache(batch_size_per_device)

    def update_snippet_list(self, key_values, doc_ids, input_ids):
        device_name = str(doc_ids.device)
        if doc_ids.size(1) > self.snippet_size:
            s = torch.randint(0, doc_ids.size(1) - self.snippet_size + 1, size=(1,)).item()
        else:
            s = 0
        
        sliced_key_values = ()
        for k, v, h, t in key_values:
            sliced_key_values = sliced_key_values + ((k[:,:, s:s+self.snippet_size].contiguous(), \
                                                      v[:,:, s:s+self.snippet_size].contiguous(), \
                                                      h[:,s:s+self.snippet_size].contiguous(),\
                                                        doc_ids[:, s:s+self.snippet_size].contiguous(), \
                                                        input_ids[:, s:s+self.snippet_size].contiguous(), t),)

        if device_name in self.snippet_list:
            if len(self.snippet_list[device_name]) == self.snip_list_len:    # need to replace
                idx = torch.randint(0, self.snip_list_len, size=(1,)).item()
                self.snippet_list[device_name][idx] = sliced_key_values
            elif len(self.snippet_list[device_name]) < self.snip_list_len:
                self.snippet_list[device_name].append(sliced_key_values)
            else:
                raise RuntimeError
        else:
            self.snippet_list[device_name] = [sliced_key_values]

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

        device_name = str(input_ids.device)
        if device_name in self.snippet_list and self.training:
            idx = torch.randint(0, len(self.snippet_list[device_name]), size=(1,)).item()
            past_key_values = self.snippet_list[device_name][idx]
        else:
            past_key_values = None

        self.timestamp[device_name] = self.timestamp.get(device_name, 0) + 1

        # Decode
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            doc_ids=doc_ids,
            past_key_values=past_key_values,
            timestamp=self.timestamp[device_name],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs.last_hidden_state
        
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

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            token_nums = (shift_labels != -100).sum()
            if token_nums.item() == 0:
                loss = torch.zeros_like(loss)
        else:
            lm_logits = None
            token_nums = None

        if self.training:
            self.update_snippet_list(decoder_outputs.present_key_values, doc_ids, input_ids)


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
