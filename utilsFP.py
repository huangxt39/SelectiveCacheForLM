import torch
from tqdm import tqdm
import pandas as pd
import time
import numpy as np
import random
import os
import pickle
from datasets import Dataset, load_from_disk
import re
from transformers import Pipeline
from matplotlib import pyplot as plt
import time
import torch.nn as nn

def load_dataset(feature, average=False, test_proportion=0.25, shuffle=True, fix_range=12, concat=False):

    if feature not in ['nFixations', 'GD', 'TRT', 'FFD', 'SFD', 'GPT', 'TRT-noD']:
        raise RuntimeError('select a valid feature')

    path = './FPdatasets/FPdata-%s-%s-test%.2f-%s-%d-%s'%\
        (feature, 'averaged' if average else '', test_proportion, 'shuffled' if shuffle else '', fix_range, 'concat' if concat else '')
    if os.path.exists(path):
        return load_from_disk(path)

    """load data"""
    s_time = time.time()
    allData = pd.read_parquet('allData.parquet')
    print('time spend on loading: ', time.time() - s_time)

    if feature in ['GD', 'GPT', 'TRT-noD']:
        allData.drop(index=allData.loc[allData['Corpus']=='Dundee'].index, inplace=True)
    # some checks
    # allData.hist(column=feature, bins=12, by='Corpus', range=(0, 800), figsize=(8,10))
    # plt.show()

    print(allData.groupby('Corpus')[feature].mean())    # do norm on corpus level or not?
    allData[feature] = allData[feature] / allData.groupby('Corpus')[feature].transform('mean') * 100
    print(allData.groupby('Corpus')[feature].mean())

    """average over subjects"""
    if average:
        grouped = allData.groupby(['Corpus','Sent_ID','Word_ID'], as_index=False, sort=False)
        df1 = grouped['Word'].agg(lambda x: pd.Series.mode(x)[0])
        cols = ['Subj_ID',  'nFixations', 'GD',  'TRT',  'FFD',  'SFD',  'GPT']
        df2 = grouped[cols].mean()
        allData = pd.concat([df1, df2[cols]], axis=1)
        print(allData)
        # print(grouped.loc[ (grouped['Subj_ID']!=12) & (grouped['Subj_ID']!=18) & (grouped['Subj_ID']!=10) & (grouped['Subj_ID']!=14)  ])
        
        # print(allData.loc[(allData['Corpus']=='Dundee') & (allData['Subj_ID']==) & (allData['Sent_ID']=='119')].to_string())

    """map the feature values to 0,1,2...11"""
    if average:
        quant = allData[feature].quantile(np.linspace(0,1,fix_range+1)[1:-1])  # 12 intervals
        # quant = [0] + quant.tolist() + [800]
        # allData.hist(column=feature, bins=quant)
        # plt.show()
        # exit()
        quant = quant.tolist()
    else:
        nonz_values = allData[feature].loc[allData[feature]>0]
        quant = nonz_values.quantile(np.linspace(0,1,fix_range)[1:-1])    # 11 intervals
        # quant = [0, 20] + quant.tolist() + [800]
        # allData.hist(column=feature, bins=quant)
        # plt.show()
        # exit()
        quant = quant.tolist()
        quant.insert(0, 1)
    print(quant)
    
    allData['target'] = 0
    for i, quantile in enumerate(quant):
        allData.loc[allData[feature] >= quantile, 'target'] = i + 1
    print(allData.head(10))

    """group by sentence"""
    X = []
    Y = []
    Sent_ID = ''
    data_dict = allData.to_dict('records')
    for row in data_dict:
        if row['Sent_ID'] != Sent_ID:
            Sent_ID = row['Sent_ID']
            X.append([])
            Y.append([])      

        word = re.sub(r'[^\x00-\x7F]', '', row['Word'])
        if word:
            X[-1].append(word)
            Y[-1].append(row['target'])
    print('number of sentences: ', len(X))
    

    dataset = Dataset.from_dict({'words': X, 'targets': Y})

    if test_proportion > 0:
        dataset = dataset.train_test_split(test_size=test_proportion)
    print(dataset)
    dataset.save_to_disk(path)
    return dataset

class DataCollatorFP:

    def __init__(self, tokenizer, padding = True, max_length = None, pad_to_multiple_of = None, return_tensors = "pt"):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
    

    def __call__(self, features):
        features_to_pad = {key: [example[key] for example in features] for key in ['input_ids', 'attention_mask']}
        batch = self.tokenizer.pad(
            features_to_pad,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        pad_num = (batch['input_ids'] == self.tokenizer.pad_token_id).sum(dim=1).tolist()
        all_targets = []
        all_spans = []
        for pad_n, example, in zip(pad_num, features):
            all_targets.extend(example['targets'])
            all_spans.extend(example['spans'])
            if pad_n > 0:
                all_targets.append(-100)
                all_spans.append(pad_n)
        batch['labels'] = torch.LongTensor(all_targets)
        batch['spans'] = torch.LongTensor(all_spans)
        assert len(batch['labels']) == len(batch['spans'])
        
        return batch
        

class FixPredictionPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        prep_kwargs = {}
        if "batch_size" in kwargs:
            prep_kwargs["batch_size"] = kwargs["batch_size"]
        forward_kwargs = {}
        if "fix_min" in kwargs:
            forward_kwargs["fix_min"] = kwargs["fix_min"]
        return prep_kwargs, forward_kwargs, {}

    def preprocess(self, inputs, batch_size=64):
        examples = inputs['text']
        model_input = {'input_ids':[], 'attention_mask':[]}
        for s in range(0, len(examples), batch_size):
            batch = self.tokenizer(examples[s:s+batch_size], padding=True, return_tensors='pt')
            model_input['input_ids'].append(batch['input_ids'])
            model_input['attention_mask'].append(batch['attention_mask'])
        return model_input

    def _forward(self, model_inputs, fix_min=0):
        outputs = []
        fix_max = self.model.fix_range.item() - 1
        print('clamp to %d - %d'%(fix_min, fix_max))
        for i in tqdm(range(len(model_inputs['input_ids']))):
            output = self.model(input_ids=model_inputs['input_ids'][i], attention_mask=model_inputs['attention_mask'][i])
            nums = model_inputs['attention_mask'][i].sum(dim=1).tolist()
            for j, num in enumerate(nums):
                
                pred = fix_min + output['logits'][j][:num] / (fix_max) * (fix_max - fix_min)
                outputs.append(torch.round(pred).clamp(min=fix_min, max=fix_max).long().tolist())
        
        return outputs

    def postprocess(self, model_outputs):
        return model_outputs


class FixPredictionPipelineGenerator(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        prep_kwargs = {}
        if "_batch_size" in kwargs:
            prep_kwargs["_batch_size"] = kwargs["_batch_size"]
            print(prep_kwargs)
        forward_kwargs = {}
        if "fix_max" in kwargs:
            forward_kwargs["fix_max"] = kwargs["fix_max"]
            print(forward_kwargs)
        return prep_kwargs, forward_kwargs, {}

    def preprocess(self, inputs, _batch_size=8):
        input_ids = inputs['input_ids']
        batch_size = _batch_size
        # print(list(map(lambda x: len(x), input_ids[-batch_size*2:])))
        generator = ( torch.tensor(input_ids[s:s+batch_size], device=self.model.device) for s in range(0, len(input_ids), batch_size) )
        # for s in tqdm(range(0, len(input_ids), batch_size)):
        #     try:
        #         torch.tensor(input_ids[s:s+batch_size], device=self.model.device) 
        #     except:
        #         print(list(map(lambda x: len(x), input_ids[s:s+batch_size])))
        return {'input_ids': generator}

    def _forward(self, model_inputs, fix_max):
        model_outputs = []
        print('clamp to %d - %d'%(0, fix_max))
        for input_ids in tqdm(model_inputs['input_ids']):
            outputs = self.model(input_ids=input_ids)
            if isinstance(outputs, dict):
                pred = outputs['logits']
            else:
                pred = outputs
            model_outputs.extend(torch.round(pred).clamp(min=0, max=fix_max).long().tolist())
        
        return model_outputs

    def postprocess(self, model_outputs):
        return model_outputs


def show_attention(attention_map, tokens):
    num = len(tokens)
    fig, ax = plt.subplots()
    fig.set_size_inches(num//2,num//2)
    ax.imshow(attention_map.numpy())

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(num), labels=tokens)
    ax.set_yticks(np.arange(num), labels=tokens)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(num):
        for j in range(num):
            text = ax.text(j, i, round(attention_map[i, j].item(), 1),
                        ha="center", va="center", color="w")

    ax.set_title("attention map")
    fig.tight_layout()
    plt.show()


class BiLSTMRegression(nn.Module):
    def __init__(self, embedding, hidden_dim, drop_out) -> None:
        super().__init__()
        self.emb = embedding
        self.emb.requires_grad_(False)

        self.lstm = nn.LSTM(input_size=self.emb.weight.size(1),
                            hidden_size=hidden_dim, 
                            num_layers=2,
                            batch_first=True,
                            dropout=drop_out,
                            bidirectional=True)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.emb(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.head(x)
        return x.squeeze(-1)


if __name__ == '__main__':
    dataset = load_dataset("TRT", True)
    print(dataset['train'][:2])