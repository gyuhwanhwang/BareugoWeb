#!/usr/bin/env python
# coding: utf-8

# In[240]:


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch.nn.functional as F
import json


# In[251]:


STRICT = 0.9
# FILE_NAME = 'test.json'


# In[252]:


data = { "0" : "되나요 되나요 시이발",
       "1" : "들리시나요 제 목소리가"}

# df = pd.DataFrame({
#  'conents' : list(data.values())
# })

# In[243]:


class Dataset_(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 0]
        return text


# In[244]:


def AI_Classifier(data):
    STRICT = 0.9
    df = pd.DataFrame({
        'conents': list(data.values())
    })

    dataset_ = Dataset_(df)
    loader = DataLoader(dataset_, shuffle=True, num_workers=0)

    # In[245]:

    # if device is gpu, using this code
    # PATH = './model/'

    # device = torch.device('cuda')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # model = torch.load(PATH + 'model.pt')
    # model.to(device)

    # if device is cpu, using this code
    PATH = './model/'
    device = torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = torch.load('C:\dev\BareugoWeb\BareugoWeb\classify\model\model.pt', map_location=device)
    model.to(device)

    # In[246]:

    model.eval()

    strong = []
    weak = []
    idx = 0

    for text in loader:
        encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
        padded_list = [e + [0] * (512 - len(e)) for e in encoded_list]
        del encoded_list

        sample = torch.tensor(padded_list)
        del padded_list

        sample = sample.to(device)
        outputs = model(sample)[0]
        value = F.softmax(outputs)
        del outputs

        pred = torch.argmax(value, dim=1)

        if int(pred) == 1:
            weak.append(idx)
        elif (int(pred) == 1) and (float(value[0][1]) > STRICT):
            strong.append(idx)

        idx += 1

    # In[247]:

    json_output = dict({
        '0': [int(s) for s in strong],
        '1': [int(w) for w in weak]
    })
    print(json_output)
    return json_output

# AI_Classifier(data)
