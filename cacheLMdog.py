from datasets import load_from_disk
from transformers import AutoTokenizer, TrainerCallback, TrainingArguments
import math
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import random
import os
import pickle
import re
from matplotlib import pyplot as plt

from newModelsT5Simplified import T5ModelDecoderCacheSelector
from newModelsT5SimplifiedRpl import T5ModelDecoderCacheReplacer
from newModelsT5SimplifiedCpr import T5ModelDecoderCacheCompressor
from utilsFP import BiLSTMRegression
from utilsLM import customTrainer, customCollator, process_dog
import logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--num_proc', type=int, default=1)
parser.add_argument('--regen_data', action='store_true')
parser.add_argument('--xl_cache_size', type=int, default=0)
parser.add_argument('--sec_cache_size', type=int, default=0)
parser.add_argument('--snippet_size', type=int, default=0)
parser.add_argument('--snip_list_len', type=int, default=0)
parser.add_argument('--L0_lambda', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--acc_steps', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--block_size', type=int, default=512)
parser.add_argument('--threshold', type=int, default=9)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model', type=str, default='small', choices=['small', 'base'])
parser.add_argument('--RSI', action='store_true')
parser.add_argument('--cache', default='slc', choices=['slc', 'rpl', 'cpr'])
parser.add_argument('--cpr_rate', type=int, default=2)

args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_name =  f"google/t5-{args.model}-lm-adapt" 
tokenizer_name = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir='./cache/models', model_max_length=2048)

if args.cache == 'slc':
    model = T5ModelDecoderCacheSelector.from_pretrained(model_name, block_size=args.block_size, xl_cache_size=args.xl_cache_size, sec_cache_size=args.sec_cache_size, \
                                    threshold=args.threshold, snippet_size=args.snippet_size, \
                                    snip_list_len=args.snip_list_len, RSI=args.RSI, L0_lambda=args.L0_lambda, tokenizer=tokenizer)
elif args.cache == 'rpl':
    model = T5ModelDecoderCacheReplacer.from_pretrained(model_name, block_size=args.block_size, xl_cache_size=args.xl_cache_size, sec_cache_size=args.sec_cache_size, \
                                     snippet_size=args.snippet_size, \
                                    snip_list_len=args.snip_list_len, tokenizer=tokenizer)
elif args.cache == 'cpr':
    model = T5ModelDecoderCacheCompressor.from_pretrained(model_name, block_size=args.block_size, xl_cache_size=args.xl_cache_size, sec_cache_size=args.sec_cache_size, \
                                     compress_rate=args.cpr_rate)


print(model_name, 'num_layers', model.config.num_decoder_layers, 'num_heads', model.config.num_heads, 'hidden_size', model.config.d_model)

block_size = args.block_size
batch_size = args.batch_size 

tokenizer.pad_token = tokenizer.eos_token

data_path = f'./LMdatasets/processed-dog3-{tokenizer_name}-{batch_size}-{block_size}-fix12'
if os.path.exists(data_path) and not args.regen_data:
    print('loading dataset...')
    tokenized_dataset = load_from_disk(data_path)
else:
    tokenized_dataset = process_dog(batch_size, block_size, tokenizer, './LMdatasets/merged-docs2')
    FP_dir = './FPmodels/T5-tokenizer-BiLSTM-TRT-12-concat-3'
    print('FP dir:', FP_dir)
    empty_emb = nn.Embedding(model.config.vocab_size, 512)
    FP_model = BiLSTMRegression(empty_emb, 128, 0.2).to(device)
    FP_model.load_state_dict(torch.load(FP_dir))
    FP_model.eval()

    print('adding fixation values...')
    for split in tokenized_dataset.keys():
        bs = batch_size if split == 'train' else 4
        input_ids = tokenized_dataset[split]['input_ids']

        fix_duration = []
        with torch.no_grad():
            for s in tqdm(range(0, len(input_ids), bs)):
                pred = FP_model(torch.tensor(input_ids[s:s+bs], device=device))
                fix_duration.extend(pred.tolist())
        
        tokenized_dataset[split] = tokenized_dataset[split].add_column('fix_duration', fix_duration)

        assert all(len(a) == len(b) for a,b in zip(tokenized_dataset[split]['input_ids'], tokenized_dataset[split]['fix_duration']))
    del FP_model
    tokenized_dataset.save_to_disk(data_path)

idx = random.randint(0, len(tokenized_dataset['train'])-1)
example = tokenized_dataset['train'][idx]
print(example)
print(' | '.join([f'{w.lstrip("Ġ")} {f:.1f}' for w,f in zip(tokenizer.convert_ids_to_tokens(example['input_ids']), example['fix_duration'])]))


print('block size: ', block_size)
print('batch size', batch_size)
print(tokenized_dataset['train'].column_names)


data_collator = customCollator(tokenizer=tokenizer)


out_dir = "experiments/experiments-%d"%(random.randint(0, 999))
while os.path.exists(out_dir):
    out_dir = "experiments/experiments-%d"%(random.randint(0, 999))

num_device = max(torch.cuda.device_count(), 1)
per_device_batch_size = batch_size // num_device
assert per_device_batch_size * num_device == batch_size

per_device_batch_size_eval = 4 // num_device
assert per_device_batch_size_eval * num_device == 4

class ResetCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, model, **kwargs):
        model.reset_cache(per_device_batch_size_eval)

    def on_epoch_begin(self, args, state, control, model, **kwargs):
        model.reset_cache(per_device_batch_size)

training_args = TrainingArguments(
    output_dir=out_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=per_device_batch_size,
    per_device_eval_batch_size=per_device_batch_size_eval,
    gradient_accumulation_steps=args.acc_steps,
    evaluation_strategy='epoch',
    num_train_epochs=30 if 'base' in model_name else 30,
    save_strategy="epoch",
    save_total_limit=3,
    logging_strategy="epoch",
    learning_rate=args.lr,
    weight_decay=0.01,
    optim='adamw_torch',
    fp16=True,
    lr_scheduler_type='constant',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    # torch_compile=True,
    # remove_unused_columns=False
)

trainer = customTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    callbacks=[ResetCallback]
)

trainer.train()

print(out_dir)
test_loader = trainer.get_test_dataloader(tokenized_dataset["test"])
with torch.no_grad():
    total_loss = 0
    total_num = 0
    trainer.model.reset_cache(per_device_batch_size_eval)
    if trainer.args.n_gpu > 1:
        model = nn.DataParallel(trainer.model)
    else:
        model = trainer.model
    model.eval()
    count = 0
    for inputs in tqdm(test_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        total_loss += (outputs.loss * outputs.token_nums).sum()
        total_num += outputs.token_nums.sum()


    print(f"test Perplexity: {math.exp( total_loss / total_num ):.2f}")
