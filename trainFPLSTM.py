from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, GPT2Config, T5Config, T5ForConditionalGeneration
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, GPT2LMHeadModel, AutoModel
from transformers import Pipeline, TextClassificationPipeline
from transformers import DataCollatorForLanguageModeling, TrainerCallback
import math
import argparse
from tqdm import tqdm
import torch
import random
import os
import pickle
import re
import torch.nn as nn
from torch.utils.data import DataLoader

from utilsFP import FixPredictionPipelineGenerator, BiLSTMRegression, load_dataset
from utilsLM import customTrainer, customCollator, process_pg19

parser = argparse.ArgumentParser()
parser.add_argument('--num_proc', type=int, default=1)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

torch.manual_seed(args.seed)

dataset = load_dataset("TRT", average=True, test_proportion=0, fix_range=12, concat=True)
save_path = './FPmodels/T5-tokenizer-BiLSTM-TRT-12-concat-3'

model_name = 't5-small'
tokenizer_name = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir='./cache/models', model_max_length=2048)

def tokenize_func(examples):
    default_target = 1
    input_words = examples['words']
    input_targets = examples['targets']
    tokenized = tokenizer([' '.join(example) for example in input_words], is_split_into_words=False, return_offsets_mapping=True)
    input_ids = tokenized['input_ids']
    offset_mapping = tokenized['offset_mapping']
    # print(len(' '.join(input_words[0])))
    # print(input_ids[0])
    # print(tokenizer.convert_ids_to_tokens(input_ids[0]))
    # print(offset_mapping[0])
    new_targets = []
    for i in range(len(input_ids)):
        char_level_targets = []
        for t, w in zip(input_targets[i], input_words[i]):
            if len(w) == 1:
                char_level_targets.extend([t, default_target])
            # elif len(w) > 1:
            #     char_level_targets.extend( [default_target,] * (len(w) - 2) + [t, default_target, default_target])
            elif len(w) > 1:
                char_level_targets.extend( [t,] * (len(w) - 1) + [default_target, default_target])
            # char_level_targets.extend([default_target,] * (len(w) - 1))
            # char_level_targets.append(t)
            # char_level_targets.append(default_target)
        char_level_targets.pop(-1)
        assert len(' '.join(input_words[i])) == len(char_level_targets)

        tokenized_targets = []
        for span in offset_mapping[i]: 
            s, e = span
            t = max(char_level_targets[s:e]) if e > s else default_target
            tokenized_targets.append(t)
        assert len(tokenized_targets) == len(input_ids[i])
        new_targets.append(tokenized_targets)

        # print([(w,t) for w, t in zip(input_words[i], input_targets[i])])
        # print([(t,token) for t, token in zip(tokenized_targets, tokenizer.convert_ids_to_tokens(input_ids[i]))])
        
    tokenized['targets'] = new_targets
    return tokenized

tokenized_dataset = dataset.map(tokenize_func, batched=True, num_proc=args.num_proc)
tokenized_dataset = tokenized_dataset.remove_columns(['words','offset_mapping', 'attention_mask'])

block_size = 512
print('block size: ', block_size)
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

tokenized_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=args.num_proc, batch_size=5000)

splitted_dataset = tokenized_dataset.train_test_split(test_size=0.25)


model = T5ForConditionalGeneration.from_pretrained(model_name)
model = BiLSTMRegression(model.shared, 128, 0.2)

def collator(examples):
    outputs = {k: torch.tensor([example[k] for example in examples]) for k in examples[0].keys() }
    return outputs

train_loader = DataLoader(splitted_dataset['train'], batch_size=args.batch_size, shuffle=True, collate_fn=collator) 
test_loader = DataLoader(splitted_dataset['test'], batch_size=args.batch_size, shuffle=False, collate_fn=collator) 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
loss_func = torch.nn.MSELoss()

best_loss = float('inf')
for epoch in range(args.num_epoch):
    model.train()
    loss_sum = []
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch['input_ids'])
        loss = loss_func(pred, batch['targets'].float())
        loss_sum.append(loss.item())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    train_loss = sum(loss_sum) / len(loss_sum)

    with torch.no_grad():
        model.eval()
        loss_sum = []
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch['input_ids'])
            loss = loss_func(pred, batch['targets'].float())
            loss_sum.append(loss.item())
        test_loss = sum(loss_sum) / len(loss_sum)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), save_path)
        print(f"epoch {epoch}: \t train loss {train_loss:.4f} \t test loss {test_loss:.4f} \t best loss {best_loss:.4f}")
