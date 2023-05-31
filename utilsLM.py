from transformers import Trainer
from transformers.training_args import TrainingArguments, OptimizerNames
from transformers.optimization import Adafactor
from transformers.utils import is_sagemaker_mp_enabled, logging
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import ShardedDDPOption
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from dataclasses import dataclass
from collections.abc import Mapping

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch.nn as nn
import torch
from distutils.util import strtobool
import os
import glob
from tqdm import tqdm
import random
import re

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.get_logger(__name__)

class customTrainer(Trainer):
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
        
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                from fairscale.optim import OSS
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum(dict((p.data_ptr(), p.numel()) for p in module.parameters()).values())
                            print(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    print(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


    def _get_train_sampler(self):
        sampler = super()._get_train_sampler()
        print(sampler)
        sampler = torch.utils.data.SequentialSampler(self.train_dataset)
        print(sampler)
        return sampler

    def get_train_dataloader(self):
        loader = super().get_train_dataloader()
        return loader
    
    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache(self.args.per_device_eval_batch_size)
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    
@dataclass
class customCollator():

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        return self.torch_call(features)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        # print(examples) # list of dicts
  
        batch = {k: torch.tensor([item[k] for item in examples]) for k in examples[0].keys()}

        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
    


def process_dog(batch_size, block_size, tokenizer, work_dir):
    
    def rearrange(file_names, batch_size):
        all_ids = []
        for file in tqdm(file_names):
            with open(file, 'r') as f:
                all_ids.append( tokenizer(f.read(), return_attention_mask=False)['input_ids'] )
        lengths = list(map(lambda x: len(x), all_ids))
        order = torch.tensor(lengths).argsort(descending=True).tolist()
            
        nums = torch.zeros(batch_size)
        indices = [[] for _ in range(batch_size)]
        for i in order:
            shortest = nums.argmin().item()
            ids = all_ids[i]
            indices[shortest].append(i)
            nums[shortest] += len(ids)
        for i in range(batch_size):
            random.shuffle(indices[i])

        input_ids = []
        doc_ids = []
        
        for i in range(batch_size):
            i_ids = []
            d_ids = []
            for j in indices[i]:
                i_ids.extend(all_ids[j])
                d_ids.extend([j,]*len(all_ids[j]))
            input_ids.append(i_ids)
            doc_ids.append(d_ids)
        print(nums)
        print(list(map(lambda x: len(x), input_ids)))
        return nums, input_ids, doc_ids

    files = glob.glob(os.path.join(work_dir, 'train', "*.txt"))

    nums, input_ids, doc_ids = rearrange(files, batch_size)

    total_seq_len = nums.min().long().item()
    data_dict = {'input_ids': [], 'doc_ids': []}
    for i in range(0, total_seq_len, block_size):
        e = min(i+block_size, total_seq_len)
        for j in range(batch_size):
            data_dict['input_ids'].append(input_ids[j][i:e])
            data_dict['doc_ids'].append(doc_ids[j][i:e])

    train_split = Dataset.from_dict(data_dict)

    # ====================
    files = glob.glob(os.path.join(work_dir, 'valid', "*.txt"))

    nums, input_ids, doc_ids = rearrange(files, 4)

    total_seq_len = nums.min().long().item()
    data_dict = {'input_ids': [], 'doc_ids': []}
    for i in range(0, total_seq_len, block_size):
        e = min(i+block_size, total_seq_len)
        for j in range(4):
            data_dict['input_ids'].append(input_ids[j][i:e])
            data_dict['doc_ids'].append(doc_ids[j][i:e])

    valid_split = Dataset.from_dict(data_dict)

    # =====================
    files = glob.glob(os.path.join(work_dir, 'test', "*.txt"))

    nums, input_ids, doc_ids = rearrange(files, 4)

    total_seq_len = nums.max().long().item()
    for i in range(4):
        pad_num = total_seq_len - len(input_ids[i])
        input_ids[i].extend([tokenizer.pad_token_id]*pad_num)
        doc_ids[i].extend([-100]*pad_num)

    data_dict = {'input_ids': [], 'doc_ids': []}
    for i in range(0, total_seq_len, block_size):
        for j in range(4):
            data_dict['input_ids'].append(input_ids[j][i:i+block_size])
            data_dict['doc_ids'].append(doc_ids[j][i:i+block_size])

    test_split = Dataset.from_dict(data_dict)

    return DatasetDict({'train': train_split, 'test': test_split, 'validation': valid_split})


def process_wiki(batch_size, block_size, tokenizer):
    wiki_data = load_dataset('wikitext', 'wikitext-2-raw-v1')

    def rearrange(subset, batch_size):
        all_ids = []
        for example in tqdm(subset):
            if example['text']:
                if len(re.findall(r"=", example['text'])) == 2:
                    all_ids.append([])
                all_ids[-1].extend( tokenizer(example['text'], return_attention_mask=False)['input_ids'] )
        lengths = list(map(lambda x: len(x), all_ids))
        order = torch.tensor(lengths).argsort(descending=True).tolist()

        nums = torch.zeros(batch_size)
        indices = [[] for _ in range(batch_size)]
        for i in order:
            shortest = nums.argmin().item()
            ids = all_ids[i]
            indices[shortest].append(i)
            nums[shortest] += len(ids)
        for i in range(batch_size):
            random.shuffle(indices[i])

        input_ids = []
        doc_ids = []
        
        for i in range(batch_size):
            i_ids = []
            d_ids = []
            for j in indices[i]:
                i_ids.extend(all_ids[j])
                d_ids.extend([j,]*len(all_ids[j]))
            input_ids.append(i_ids)
            doc_ids.append(d_ids)
        print(nums)
        print(list(map(lambda x: len(x), input_ids)))
        return nums, input_ids, doc_ids
    
    nums, input_ids, doc_ids = rearrange(wiki_data['train'], batch_size)

    total_seq_len = nums.min().long().item()
    data_dict = {'input_ids': [], 'doc_ids': []}
    for i in range(0, total_seq_len, block_size):
        e = min(i+block_size, total_seq_len)
        for j in range(batch_size):
            data_dict['input_ids'].append(input_ids[j][i:e])
            data_dict['doc_ids'].append(doc_ids[j][i:e])

    train_split = Dataset.from_dict(data_dict)

    # ====================
    nums, input_ids, doc_ids = rearrange(wiki_data['validation'], 8)

    total_seq_len = nums.min().long().item()
    data_dict = {'input_ids': [], 'doc_ids': []}
    for i in range(0, total_seq_len, block_size):
        e = min(i+block_size, total_seq_len)
        for j in range(8):
            data_dict['input_ids'].append(input_ids[j][i:e])
            data_dict['doc_ids'].append(doc_ids[j][i:e])

    valid_split = Dataset.from_dict(data_dict)

    # =====================
    nums, input_ids, doc_ids = rearrange(wiki_data['test'], 8)

    total_seq_len = nums.max().long().item()
    for i in range(8):
        pad_num = total_seq_len - len(input_ids[i])
        input_ids[i].extend([tokenizer.pad_token_id]*pad_num)
        doc_ids[i].extend([-100]*pad_num)

    data_dict = {'input_ids': [], 'doc_ids': []}
    for i in range(0, total_seq_len, block_size):
        for j in range(8):
            data_dict['input_ids'].append(input_ids[j][i:i+block_size])
            data_dict['doc_ids'].append(doc_ids[j][i:i+block_size])

    test_split = Dataset.from_dict(data_dict)

    return DatasetDict({'train': train_split, 'test': test_split, 'validation': valid_split})


def process_pg19(batch_size, block_size, tokenizer, work_dir):
    
    def rearrange(file_names, batch_size):
        all_ids = []
        for file in tqdm(file_names):
            with open(file, 'r') as f:
                all_ids.append( tokenizer(f.read(), return_attention_mask=False)['input_ids'] )
        lengths = list(map(lambda x: len(x), all_ids))
        order = torch.tensor(lengths).argsort(descending=True).tolist()
            
        nums = torch.zeros(batch_size)
        indices = [[] for _ in range(batch_size)]
        for i in order:
            shortest = nums.argmin().item()
            ids = all_ids[i]
            indices[shortest].append(i)
            nums[shortest] += len(ids)
        for i in range(batch_size):
            random.shuffle(indices[i])

        input_ids = []
        doc_ids = []
        
        for i in range(batch_size):
            i_ids = []
            d_ids = []
            for j in indices[i]:
                i_ids.extend(all_ids[j])
                d_ids.extend([j,]*len(all_ids[j]))
            input_ids.append(i_ids)
            doc_ids.append(d_ids)
        print(nums)
        print(list(map(lambda x: len(x), input_ids)))
        return nums, input_ids, doc_ids

    with open(os.path.join(work_dir, 'train_subset.txt'), 'r') as f:
        files = f.read().split('\n')
    files = list(map(lambda f: os.path.join(work_dir, 'train', f), files))

    nums, input_ids, doc_ids = rearrange(files, batch_size)

    total_seq_len = nums.min().long().item()
    data_dict = {'input_ids': [], 'doc_ids': []}
    for i in range(0, total_seq_len, block_size):
        e = min(i+block_size, total_seq_len)
        for j in range(batch_size):
            data_dict['input_ids'].append(input_ids[j][i:e])
            data_dict['doc_ids'].append(doc_ids[j][i:e])

    train_split = Dataset.from_dict(data_dict)

    # ====================
    files = glob.glob(os.path.join(work_dir, 'validation', "*.txt"))

    nums, input_ids, doc_ids = rearrange(files, 8)

    total_seq_len = nums.min().long().item()
    data_dict = {'input_ids': [], 'doc_ids': []}
    for i in range(0, total_seq_len, block_size):
        e = min(i+block_size, total_seq_len)
        for j in range(8):
            data_dict['input_ids'].append(input_ids[j][i:e])
            data_dict['doc_ids'].append(doc_ids[j][i:e])

    valid_split = Dataset.from_dict(data_dict)

    # =====================
    files = glob.glob(os.path.join(work_dir, 'test', "*.txt"))

    nums, input_ids, doc_ids = rearrange(files, 8)

    total_seq_len = nums.max().long().item()
    for i in range(8):
        pad_num = total_seq_len - len(input_ids[i])
        input_ids[i].extend([tokenizer.pad_token_id]*pad_num)
        doc_ids[i].extend([-100]*pad_num)

    data_dict = {'input_ids': [], 'doc_ids': []}
    for i in range(0, total_seq_len, block_size):
        for j in range(8):
            data_dict['input_ids'].append(input_ids[j][i:i+block_size])
            data_dict['doc_ids'].append(doc_ids[j][i:i+block_size])

    test_split = Dataset.from_dict(data_dict)

    return DatasetDict({'train': train_split, 'test': test_split, 'validation': valid_split})

