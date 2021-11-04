# 1. Introduction

GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset of 8 million web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some text. The diversity of the dataset causes this simple goal to contain naturally occurring demonstrations of many tasks across diverse domains. GPT-2 is a direct scale-up of GPT, with more than 10X the parameters and trained on more than 10X the amount of data.

**Zero-Shot Transfer**: The pre-training task for GPT-2 is solely language modeling. All the downstream language tasks are framed as predicting conditional probabilities and there is no task-specific fine-tuning.

In this notebook, I use the dataset of writing prompts and stories from [this paper](https://github.com/pytorch/fairseq/tree/master/examples/stories) to fine-tune GPT-2, then use the fine-tuned model to generate stories. I use [perplexity](https://en.wikipedia.org/wiki/Perplexity) as the metrics to check if fine-tuning imporves the performance or not.  

![image.png](attachment:image.png)


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import logging
from tqdm import tqdm
import math
import argparse
import os
```

## 1.1 Pacakges and frameworks

**Environment**: Kaggle Python 3 environment and GPU

**Deep learning framework**: Pytorch

**NLP Package**: Transformers 3.0.2


```python
!git clone https://github.com/huggingface/transformers
!pip install transformers/
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
```

    Cloning into 'transformers'...
    remote: Enumerating objects: 89, done.[K
    remote: Counting objects: 100% (89/89), done.[K
    remote: Compressing objects: 100% (59/59), done.[K
    remote: Total 37905 (delta 42), reused 50 (delta 18), pack-reused 37816[K
    Receiving objects: 100% (37905/37905), 27.68 MiB | 25.81 MiB/s, done.
    Resolving deltas: 100% (26234/26234), done.
    Processing ./transformers
    Requirement already satisfied: numpy in /opt/conda/lib/python3.7/site-packages (from transformers==3.0.2) (1.18.5)
    Collecting tokenizers==0.8.1.rc2
      Downloading tokenizers-0.8.1rc2-cp37-cp37m-manylinux1_x86_64.whl (3.0 MB)
    [K     |████████████████████████████████| 3.0 MB 2.9 MB/s 
    [?25hRequirement already satisfied: packaging in /opt/conda/lib/python3.7/site-packages (from transformers==3.0.2) (20.1)
    Requirement already satisfied: filelock in /opt/conda/lib/python3.7/site-packages (from transformers==3.0.2) (3.0.10)
    Requirement already satisfied: requests in /opt/conda/lib/python3.7/site-packages (from transformers==3.0.2) (2.23.0)
    Requirement already satisfied: tqdm>=4.27 in /opt/conda/lib/python3.7/site-packages (from transformers==3.0.2) (4.45.0)
    Requirement already satisfied: regex!=2019.12.17 in /opt/conda/lib/python3.7/site-packages (from transformers==3.0.2) (2020.4.4)
    Requirement already satisfied: sentencepiece!=0.1.92 in /opt/conda/lib/python3.7/site-packages (from transformers==3.0.2) (0.1.91)
    Requirement already satisfied: sacremoses in /opt/conda/lib/python3.7/site-packages (from transformers==3.0.2) (0.0.43)
    Requirement already satisfied: six in /opt/conda/lib/python3.7/site-packages (from packaging->transformers==3.0.2) (1.14.0)
    Requirement already satisfied: pyparsing>=2.0.2 in /opt/conda/lib/python3.7/site-packages (from packaging->transformers==3.0.2) (2.4.7)
    Requirement already satisfied: idna<3,>=2.5 in /opt/conda/lib/python3.7/site-packages (from requests->transformers==3.0.2) (2.9)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/lib/python3.7/site-packages (from requests->transformers==3.0.2) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/lib/python3.7/site-packages (from requests->transformers==3.0.2) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.7/site-packages (from requests->transformers==3.0.2) (2020.6.20)
    Requirement already satisfied: click in /opt/conda/lib/python3.7/site-packages (from sacremoses->transformers==3.0.2) (7.1.1)
    Requirement already satisfied: joblib in /opt/conda/lib/python3.7/site-packages (from sacremoses->transformers==3.0.2) (0.14.1)
    Building wheels for collected packages: transformers
      Building wheel for transformers (setup.py) ... [?25l- \ | / - done
    [?25h  Created wheel for transformers: filename=transformers-3.0.2-py3-none-any.whl size=854680 sha256=76da88e75bc94b2dcaf743ae3a56f92f3899bec102fc679479264a340cdc44f0
      Stored in directory: /tmp/pip-ephem-wheel-cache-aeb3aaud/wheels/be/1e/28/7186a3baa6fcb4e9201f390b70b4e6d75651e85d4e8a9ae413
    Successfully built transformers
    Installing collected packages: tokenizers, transformers
      Attempting uninstall: tokenizers
        Found existing installation: tokenizers 0.7.0
        Uninstalling tokenizers-0.7.0:
          Successfully uninstalled tokenizers-0.7.0
      Attempting uninstall: transformers
        Found existing installation: transformers 2.11.0
        Uninstalling transformers-2.11.0:
          Successfully uninstalled transformers-2.11.0
    [31mERROR: After October 2020 you may experience errors when installing or updating packages. This is because pip will change the way that it resolves dependency conflicts.
    
    We recommend you use --use-feature=2020-resolver to test your packages with the new resolver before it becomes the default.
    
    allennlp 1.0.0 requires transformers<2.12,>=2.9, but you'll have transformers 3.0.2 which is incompatible.[0m
    Successfully installed tokenizers-0.8.1rc2 transformers-3.0.2
    [33mWARNING: You are using pip version 20.2.1; however, version 20.2.2 is available.
    You should consider upgrading via the '/opt/conda/bin/python3.7 -m pip install --upgrade pip' command.[0m


    [34m[1mwandb[0m: [33mWARNING[0m W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.
    

## 1.2 Arguments


```python
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=88888)
parser.add_argument("--model_name", default="gpt2", type=str)
parser.add_argument("--max_seq_length", default=512, type=int)
parser.add_argument("--train_batch_size", default=4, type=int)
parser.add_argument("--valid_batch_size", default=4, type=int)
parser.add_argument("--num_train_epochs", default=1, type=int)
parser.add_argument("--warmup", default=0.1, type=float)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--input_text_path", default='../input/story-text', type=str)
args, _ = parser.parse_known_args()
```

# 2. Prepare the data

## 2.1 Combine the prompt and story, do a little text clean
Download the text file from the above link. There are train, valid and test dataset in the original dataset. And the prompts and stories are in seperate files. For a example, the valid.wp_source has the writing promots and valid.wp_target has the corresponding stories. The train dataset is very large. Since kaggle notebook limits the kernel running time to 3 hours. I decide to take the valid dataset as my train dataset, and the test dataset as valid dataset. 

In order to feed the prompt an story together to GPT-2, I combine the prompts and stories togeter.Thus every line in the combined file includes the prompt and it's corresponding story.


```python
DATAPATH=args.input_text_path
def combinetext(prompt, story):
    fp=open(os.path.join(DATAPATH,prompt),encoding='utf8')
    fs=open(os.path.join(DATAPATH,story),encoding='utf8')
    prompts=fp.readlines()
    stories=fs.readlines()
    assert len(prompts)==len(stories)
    combine=[]
    for i in range(len(prompts)):
        combine.append(prompts[i].rstrip()+' <sep> '+" ".join(stories[i].split()[:300]))
    return combine

#do a littel text clean with punctuations
def cleanpunctuation(s):
    for p in '!,.:;?':
        s=s.replace(' '+p,p)
    s=s.replace(' '+'n\'t','n\'t')
    s=s.replace(' '+'\'s','\'s')
    s=s.replace(' '+'\'re','\'re')
    s=s.replace(' '+'\'ve','\'ve')
    s=s.replace(' '+'\'ll','\'ll')
    s=s.replace(' '+'\'am','\'am')
    s=s.replace(' '+'\'m','\'m')
    s=s.replace(' '+'\' m','\'m')
    s=s.replace(' '+'\'m','\'m')
    s=s.replace(' '+'\' ve','\'ve')
    s=s.replace(' '+'\' s','\'s')
    s=s.replace('<newline>','\n')
    return s   

train_text=combinetext('valid.wp_source', 'valid.wp_target')
train_text=list(map(cleanpunctuation,train_text))
valid_text=combinetext('test.wp_source', 'test.wp_target')
valid_text=list(map(cleanpunctuation,valid_text))
```

The below is an example of a prompt and story. 


```python
train_text[6]
```




    "[ WP ] Everyone in the world has magic with various levels of mastery over it. You are extremely powerful with almost no control so you find a demon that's very weak but extremely good at controlling his powers. <sep> `` Imagine you're in a field. '' Green extends in all directions. `` You're alone, the earth is flat, and the blue sky touches the horizon. '' Blue shoots from the ground, arcing overhead. `` The sun appears, tiny in the sky. '' There's a bright light, rays casting shadow behind me. `` What color is it? '' \n \n `` Yellow. '' It burns so brightly, winking playfully. \n \n `` Good. '' She licks her chapped lips, the sound distorting my tiny sun's light. `` Look ahead of you. There's a sheep. '' Something soft and downy wanders across the green, its shadow stretching far beyond the horizon. `` What color is it? '' \n \n My brows crease. `` Uh- '' \n \n `` What color is it? '' \n \n The green wavers. Baa baa black sheep, have you any wool? `` Uh. '' Mary had a little lamb, its fleece as white as snow. `` Um. '' The sheep wavers, cotton fuzz shifting from dark to bright as its shadows remain inky. \n \n `` What color is the sheep, Jess? '' \n \n Black and white race across downy fluff, one after the other, again and again. `` Um. '' I see the black sheep, blending into its shadow. `` Um. '' In the same blink of an eye, that sheep is white and contrasted against the verdant grass. The term verdant applies here because I'm in the countryside, you see."



## 2.2 Tokenize and load to dataloader

GPT-2 uses BPE to tokenize the text squence.BPE merges frequently co-occurred byte pairs in a greedy manner. In order to let the sequences in the same batch have the same length, I set the max length of sequence as 512, and truncate the longer sequence and pad the shorter sequence. Since the tokenizer function only return the input_ids and attention_mask. For training purpose, I need to feed the labels(targets) to the model. So I create labels sequence for every input_ids squence. In the label sequence,I rule out the padding tokens by set it to -100 to avoid compute loss on them. And also GPT-2 will automatically shift the labels to the right to match the inputs_ids, so I don't need to deal with it.



```python
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token=tokenizer.eos_token

inputs_train = tokenizer(train_text, padding=True,truncation=True,max_length=args.max_seq_length)
inputs_valid=tokenizer(valid_text, padding=True,truncation=True,max_length=args.max_seq_length)
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=1042301.0, style=ProgressStyle(descript…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=456318.0, style=ProgressStyle(descripti…


    
    


```python
def create_labels(inputs):
    labels=[]
    for ids,attention_mask in zip(inputs['input_ids'],inputs['attention_mask']):
        label=ids.copy()
        real_len=sum(attention_mask)
        padding_len=len(attention_mask)-sum(attention_mask)
        label[:]=label[:real_len]+[-100]*padding_len
        labels.append(label)
    inputs['labels']=labels
    
create_labels(inputs_train)
create_labels(inputs_valid)

```

An example of sequence of input_ids, attention_mask and labels.


```python
print(inputs_train['input_ids'][6])
print(inputs_train['attention_mask'][6])
print(inputs_train['labels'][6])

```

    [58, 28993, 2361, 11075, 287, 262, 995, 468, 5536, 351, 2972, 2974, 286, 30677, 625, 340, 13, 921, 389, 4457, 3665, 351, 2048, 645, 1630, 523, 345, 1064, 257, 3222, 326, 338, 845, 4939, 475, 4457, 922, 379, 12755, 465, 5635, 13, 1279, 325, 79, 29, 7559, 18450, 345, 821, 287, 257, 2214, 13, 10148, 3469, 14582, 287, 477, 11678, 13, 7559, 921, 821, 3436, 11, 262, 4534, 318, 6228, 11, 290, 262, 4171, 6766, 18105, 262, 17810, 13, 10148, 4518, 20611, 422, 262, 2323, 11, 610, 2259, 16965, 13, 7559, 383, 4252, 3568, 11, 7009, 287, 262, 6766, 13, 10148, 1318, 338, 257, 6016, 1657, 11, 24823, 13092, 9082, 2157, 502, 13, 7559, 1867, 3124, 318, 340, 30, 10148, 220, 198, 220, 198, 7559, 12550, 13, 10148, 632, 20246, 523, 35254, 11, 266, 8040, 711, 2759, 13, 220, 198, 220, 198, 7559, 4599, 13, 10148, 1375, 300, 3378, 607, 442, 6320, 11914, 11, 262, 2128, 1233, 24707, 616, 7009, 4252, 338, 1657, 13, 7559, 6803, 4058, 286, 345, 13, 1318, 338, 257, 15900, 13, 10148, 13742, 2705, 290, 866, 88, 11569, 364, 1973, 262, 4077, 11, 663, 9082, 20880, 1290, 3675, 262, 17810, 13, 7559, 1867, 3124, 318, 340, 30, 10148, 220, 198, 220, 198, 2011, 4772, 82, 1126, 589, 13, 7559, 28574, 12, 10148, 220, 198, 220, 198, 7559, 1867, 3124, 318, 340, 30, 10148, 220, 198, 220, 198, 383, 4077, 2082, 690, 13, 347, 7252, 275, 7252, 2042, 15900, 11, 423, 345, 597, 25749, 30, 7559, 28574, 13, 10148, 5335, 550, 257, 1310, 19343, 11, 663, 11562, 344, 355, 2330, 355, 6729, 13, 7559, 21039, 13, 10148, 383, 15900, 2082, 690, 11, 15985, 26080, 15852, 422, 3223, 284, 6016, 355, 663, 16187, 3520, 287, 2584, 13, 220, 198, 220, 198, 7559, 1867, 3124, 318, 262, 15900, 11, 12707, 30, 10148, 220, 198, 220, 198, 2619, 290, 2330, 3234, 1973, 866, 88, 781, 1648, 11, 530, 706, 262, 584, 11, 757, 290, 757, 13, 7559, 21039, 13, 10148, 314, 766, 262, 2042, 15900, 11, 34863, 656, 663, 9082, 13, 7559, 21039, 13, 10148, 554, 262, 976, 21019, 286, 281, 4151, 11, 326, 15900, 318, 2330, 290, 49754, 1028, 262, 3326, 67, 415, 8701, 13, 383, 3381, 3326, 67, 415, 8991, 994, 780, 314, 1101, 287, 262, 25708, 11, 345, 766, 13, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256]
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [58, 28993, 2361, 11075, 287, 262, 995, 468, 5536, 351, 2972, 2974, 286, 30677, 625, 340, 13, 921, 389, 4457, 3665, 351, 2048, 645, 1630, 523, 345, 1064, 257, 3222, 326, 338, 845, 4939, 475, 4457, 922, 379, 12755, 465, 5635, 13, 1279, 325, 79, 29, 7559, 18450, 345, 821, 287, 257, 2214, 13, 10148, 3469, 14582, 287, 477, 11678, 13, 7559, 921, 821, 3436, 11, 262, 4534, 318, 6228, 11, 290, 262, 4171, 6766, 18105, 262, 17810, 13, 10148, 4518, 20611, 422, 262, 2323, 11, 610, 2259, 16965, 13, 7559, 383, 4252, 3568, 11, 7009, 287, 262, 6766, 13, 10148, 1318, 338, 257, 6016, 1657, 11, 24823, 13092, 9082, 2157, 502, 13, 7559, 1867, 3124, 318, 340, 30, 10148, 220, 198, 220, 198, 7559, 12550, 13, 10148, 632, 20246, 523, 35254, 11, 266, 8040, 711, 2759, 13, 220, 198, 220, 198, 7559, 4599, 13, 10148, 1375, 300, 3378, 607, 442, 6320, 11914, 11, 262, 2128, 1233, 24707, 616, 7009, 4252, 338, 1657, 13, 7559, 6803, 4058, 286, 345, 13, 1318, 338, 257, 15900, 13, 10148, 13742, 2705, 290, 866, 88, 11569, 364, 1973, 262, 4077, 11, 663, 9082, 20880, 1290, 3675, 262, 17810, 13, 7559, 1867, 3124, 318, 340, 30, 10148, 220, 198, 220, 198, 2011, 4772, 82, 1126, 589, 13, 7559, 28574, 12, 10148, 220, 198, 220, 198, 7559, 1867, 3124, 318, 340, 30, 10148, 220, 198, 220, 198, 383, 4077, 2082, 690, 13, 347, 7252, 275, 7252, 2042, 15900, 11, 423, 345, 597, 25749, 30, 7559, 28574, 13, 10148, 5335, 550, 257, 1310, 19343, 11, 663, 11562, 344, 355, 2330, 355, 6729, 13, 7559, 21039, 13, 10148, 383, 15900, 2082, 690, 11, 15985, 26080, 15852, 422, 3223, 284, 6016, 355, 663, 16187, 3520, 287, 2584, 13, 220, 198, 220, 198, 7559, 1867, 3124, 318, 262, 15900, 11, 12707, 30, 10148, 220, 198, 220, 198, 2619, 290, 2330, 3234, 1973, 866, 88, 781, 1648, 11, 530, 706, 262, 584, 11, 757, 290, 757, 13, 7559, 21039, 13, 10148, 314, 766, 262, 2042, 15900, 11, 34863, 656, 663, 9082, 13, 7559, 21039, 13, 10148, 554, 262, 976, 21019, 286, 281, 4151, 11, 326, 15900, 318, 2330, 290, 49754, 1028, 262, 3326, 67, 415, 8701, 13, 383, 3381, 3326, 67, 415, 8991, 994, 780, 314, 1101, 287, 262, 25708, 11, 345, 766, 13, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
    


```python
class StoryDataset:
    def __init__(self, inputs):
        self.ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.labels=inputs['labels']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):

        return [torch.tensor(self.ids[item], dtype=torch.long),
                torch.tensor(self.attention_mask[item], dtype=torch.long),
                torch.tensor(self.labels[item], dtype=torch.long)]
            
```


```python
train_batch_size=args.train_batch_size
valid_batch_size=args.valid_batch_size
traindata=StoryDataset(inputs_train)
train_dataloader = torch.utils.data.DataLoader(
    traindata,
    shuffle=False,
    batch_size=train_batch_size)

validdata=StoryDataset(inputs_valid)
valid_dataloader = torch.utils.data.DataLoader(
    validdata,
    shuffle=False,
    batch_size=valid_batch_size)
```

# 3. Model and optimizer

## 3.1 Zero-shot story generate

With the amazing transfomers pacakge, we can easily download the pretrained GPT-2 model. **Before fine-tuning the model, I evaluate the model with valid dataset, and the average perplexity of evaluate results is 39**. Let's see what is the score of perplexity after fine-tuning later.


```python
model = GPT2LMHeadModel.from_pretrained('gpt2')
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=665.0, style=ProgressStyle(description_…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=548118077.0, style=ProgressStyle(descri…


    
    


```python
model.to('cuda')
model.eval()
eval_loss=[]
for inputs in tqdm(valid_dataloader, desc="eval"):
    d1,d2,d3=inputs
    d1=d1.to('cuda')        
    d2=d2.to('cuda')
    d3=d3.to('cuda')

    with torch.no_grad():
        output = model(input_ids=d1, attention_mask=d2,labels=d3)
        batch_loss=output[0]
    eval_loss+=[batch_loss.cpu().item()]
    del batch_loss
eval_loss=np.mean(eval_loss)
perplexity=math.exp(eval_loss)
print(f'The average perplexity for valid dataset before fine-tuning is {perplexity}') 
```

    eval: 100%|██████████| 3785/3785 [06:57<00:00,  9.08it/s]

    The average perplexity for valid dataset before fine-tuning is 39.27880551765438
    

    
    

Let's pick a prompt from the valid dataset and input it into the model, have the model generate a 300 words long story. The output stories is really great!  I use the generate method comes with the model. The method currently supports greedy decoding, beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling. The meanings of key arguments are below:<br> 
1)**do_sample**: if set to False greedy decoding is used.<br>
2)The **temperature** is used to module the next token probabilities.<br>
3)**top_k** is the number of highest probability vocabulary tokens to keep for top-k-filtering.<br>
4)**top_p** is the cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. <br>5)**repetition_penalty** is the parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty.<br>


```python
prompt=valid_text[300][:valid_text[300].find('<sep>')]
target=valid_text[300][valid_text[300].find('<sep>')+5:]

def generate_story(prompt,target,k=0,p=0.9,output_length=300,temperature=1,num_return_sequences=3,repetition_penalty=1.0):
    print("====prompt====\n")
    print(prompt+"\n")
    print('====target story is as below===\n')
    print(target+"\n")
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    model.to('cpu')
    model.eval()
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=output_length,
        temperature=temperature,
        top_k=k,
        top_p=p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=num_return_sequences
    )
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()
        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        # Remove all text after eos token
        text = text[: text.find(tokenizer.eos_token)]
        print(text)

generate_story(prompt,target)
```

    ====prompt====
    
    Children's logic dictates the way the world works. [ WP ] 
    
    ====target story is as below===
    
     “ That ’ s not an option I ’ m currently willing to exercise. ” 
     
     I pinch the bridge of my nose to stave off the headache building behind my eyes. If this goes on much longer, I ’ m gon na have to start to start cutting back on the vegetables. 
     
     “ She ’ s dangerous, Jimmy. You know that. You ’ ve seen it. Dealt with it first hand. She just doesn ’ t play by anyone ’ s rules. ” 
     
     Ali finished off her sucker and unwrapped a fresh one, offering it to me. I declined. I ’ d sworn off the things after my third cavity scare. That one saw me at the dentist for the third time in as many months. I don ’ t care what my dad says, I know that guy is evil. Who owns a drill like that? A murderer, that ’ s who. I still hear the damn thing in my nightmares. 
     
     While she savored the smooth flavor of blue-raspberry, I pondered her words. We both knew she was right. The situation was spiralling out of control. The details of our latest reports flooded my mind, even as I sing songed a few La-La ’ s to block them out. 
     
     Our perp is a wildcard and damn near untraceable. Hide and seek skills like none I ’ ve ever seen. In another life she might have made the perfect detective, but as it was, no one could trust her as far as they could throw her, and she always went
    
    === GENERATED SEQUENCE 1 ===
    Children's logic dictates the way the world works. [ WP ] __________________________________________
    
    12. Medical education
    
    What is most interesting about our history of education is the rate at which graduates commit suicide. Are all these leaders fighting "the system's delusions"?
    
    What does this show, anyway? Let us know in the comments section below or on Facebook.
    === GENERATED SEQUENCE 2 ===
    Children's logic dictates the way the world works. [ WP ]  "You want to make sure you're smart, because you will be able to control your own behavior."    "You want to lead a safe, happy life."    "You want to be strong for kids."    "You want to be a part of a big team who keeps the boat running smoothly."    "Your awareness will go a long way to resolving errors, all of which are critical to success."  "You will make a huge difference to our neighbors' lives."    "Your talent, your honesty, will pay dividends for us."    "Your support will fund our businesses, and lead us into a better place."    "Your vision will be incredibly influential in our lives."    "Your kindness, strength and authenticity will be decisive in how we might manage the world."    "You will make a difference for your children."    "You will look up to the biggest people in the world."    "You will create a positive sense of community."    "You will be a role model for future generations."    "You will live a life that is collaborative."    "You will not shy away from talk to grow," he wrote in Men Among Us. [ WP ]  "There is no such thing as elite status. Self-worth depend
    === GENERATED SEQUENCE 3 ===
    Children's logic dictates the way the world works. [ WP ]   ~~~~~~ __ k I ON WORLD ROLE __ k
    

## 3.1 Fine-tune the model

The number of training samples is 15620. With one GPU to train the model, it tooks about 21 minutes to run 1 epoch. **After 1 epoche learning, the perplexity for valid dataset is about 24**, which is better than the score before fine- tuning.


```python
num_train_epochs = args.num_train_epochs
training_steps_per_epoch=len(train_dataloader)
total_num_training_steps = int(training_steps_per_epoch*num_train_epochs)
weight_decay=0
learning_rate=args.learning_rate
adam_epsilon=1e-8
warmup_steps=int(total_num_training_steps*args.warmup)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_num_training_steps
)
```


```python
print("***** Running training *****")
print("  Total_num_training_step = {}".format(total_num_training_steps))
print("  Num Epochs = {}".format(num_train_epochs))
print(f"  Train_batch_size per device = {train_batch_size}")
print(f"  Valid_batch_size per device = {valid_batch_size}")
model.to('cuda')
for epoch in range(num_train_epochs):
    print(f"Start epoch{epoch+1} of {num_train_epochs}")
    train_loss=0
    epoch_iterator = tqdm(train_dataloader,desc='Iteration')
    model.train()
    model.zero_grad()    
    for _, inputs in enumerate(epoch_iterator):        
        d1,d2,d3=inputs
        d1=d1.to('cuda')
        d2=d2.to('cuda')
        d3=d3.to('cuda')
        output = model(input_ids=d1, attention_mask=d2,labels=d3)
        batch_loss=output[0]
        batch_loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        train_loss+=batch_loss.item()
        epoch_iterator.set_description('(batch loss=%g)' % batch_loss.item())
        del batch_loss
    print(f'Average train loss per example={train_loss/training_steps_per_epoch} in epoch{epoch+1}')    
    print(f'Starting evaluate after epoch {epoch+1}')
    eval_loss=[]    
    model.eval()    
    for inputs in tqdm(valid_dataloader, desc="eval"):
        d1,d2,d3=inputs
        d1=d1.to('cuda')        
        d2=d2.to('cuda')
        d3=d3.to('cuda')
        with torch.no_grad():
            output = model(input_ids=d1, attention_mask=d2,labels=d3)
            batch_loss=output[0]
        eval_loss+=[batch_loss.cpu().item()]
        del batch_loss
    eval_loss=np.mean(eval_loss)
    perplexity=math.exp(eval_loss)
    print(f'Average valid loss per example={eval_loss} in epoch{epoch+1}')    
    print(f'Perplextiy for valid dataset in epoch{epoch+1} is {perplexity}')
    
```

    Iteration:   0%|          | 0/3905 [00:00<?, ?it/s]

    ***** Running training *****
      Total_num_training_step = 3905
      Num Epochs = 1
      Train_batch_size per device = 4
      Valid_batch_size per device = 4
    Start epoch1 of 1
    

    (batch loss=2.85162): 100%|██████████| 3905/3905 [21:54<00:00,  2.97it/s]
    eval:   0%|          | 1/3785 [00:00<06:55,  9.11it/s]

    Average train loss per example=3.284638887788819 in epoch1
    Starting evaluate after epoch 1
    

    eval: 100%|██████████| 3785/3785 [06:57<00:00,  9.07it/s]

    Average valid loss per example=3.1831122323417915 in epoch1
    Perplextiy for valid dataset in epoch1 is 24.1217092154025
    

    
    

## 3.2 Generate stories
I use the fine-tuened model to generate stories with the same prompt I used before fine-tuning.


```python
prompt=valid_text[300][:valid_text[300].find('<sep>')]
target=valid_text[300][valid_text[300].find('<sep>')+5:]
generate_story(prompt,target)
```

    ====prompt====
    
    Children's logic dictates the way the world works. [ WP ] 
    
    ====target story is as below===
    
     “ That ’ s not an option I ’ m currently willing to exercise. ” 
     
     I pinch the bridge of my nose to stave off the headache building behind my eyes. If this goes on much longer, I ’ m gon na have to start to start cutting back on the vegetables. 
     
     “ She ’ s dangerous, Jimmy. You know that. You ’ ve seen it. Dealt with it first hand. She just doesn ’ t play by anyone ’ s rules. ” 
     
     Ali finished off her sucker and unwrapped a fresh one, offering it to me. I declined. I ’ d sworn off the things after my third cavity scare. That one saw me at the dentist for the third time in as many months. I don ’ t care what my dad says, I know that guy is evil. Who owns a drill like that? A murderer, that ’ s who. I still hear the damn thing in my nightmares. 
     
     While she savored the smooth flavor of blue-raspberry, I pondered her words. We both knew she was right. The situation was spiralling out of control. The details of our latest reports flooded my mind, even as I sing songed a few La-La ’ s to block them out. 
     
     Our perp is a wildcard and damn near untraceable. Hide and seek skills like none I ’ ve ever seen. In another life she might have made the perfect detective, but as it was, no one could trust her as far as they could throw her, and she always went
    
    === GENERATED SEQUENCE 1 ===
    Children's logic dictates the way the world works. [ WP ] 
     “ Mother, if I'm wrong about this, all of our children ’ s children will be born with the exact same gene that you ’ ve taken from them. ” 
     
     My six grandkids stand, the delicate delicate, clean-cut fellow in the corner and the hardy, lovely blonde cuddled under her arms. Their eyes widen at the prospect of these different choices. A sound in their ears comes, as if several big white engines chirp loudly from the mass. I look over at Mom ’ s face. Her name is Albert and she ’ d spent all summer with me. 
     
     “ Hello mom ’ s baby. Can you name some of the colors that you want to have made my grandkids? ” 
     
     “ Sun is white. May be blue or gray. Green or white, dye if you like. Maybe green, yellow. Green, yellow, blue, or brown. Blue or yellow. Whichever color you want. Make me blue! ” 
     
     “ Yes, yes, you can make me blue, Mom ’ s baby. If you want your children to be born with those same genes, please tell me where they will come from. Let me know what color it is. ” 
     
     “ Okay, she ’ d be blue
    === GENERATED SEQUENCE 2 ===
    Children's logic dictates the way the world works. [ WP ] 
     Today, you are an observer in the world's newest post-apocalyptic world. You see the smoke rising from the chimneys and discovering buildings in the distance. You are a hero, but do you really think you are alone? There is a source of constant mystery, but you don't know that. 
     
     You begin to describe an alien civilization you have no way of following. It seems like nothing more than the fault of a single soldier in that warzone. The men and women are called out and punished by their superiors, but even they are killed or dragged into service. You are the first to witness such a genocide, and you see those uncaring eyes of yours begin to deal with every detail. You feel your heart soar and you begin to take comfort in the fact that something is better than what you were expecting, but it is never far away. You are called into battle and the fighting begins. 
     
     When a tank and an armored vehicle approaches the camp you see two men standing before the camera. The men attack and fire upon them but soon the tanks fall to the ground and you scream. 
     
     `` I... I'm not really here! '' 
     
     The two men hear you curse out a few words. `` I swear you killed me, '' one of the more colorful of them adds. `` What... What did I just say? '
    === GENERATED SEQUENCE 3 ===
    Children's logic dictates the way the world works. [ WP ] 
     - Logical consequences of the crisis in the year 2078. 
     
     Humans were doing a great thing, we didn't have any problems, and we maintained the standard of living and standards in the world. We became more efficient and we won many war. The human population was at its peak after our intervention and for this reason we have discovered the way to prevent war. We hope for a peaceful resolution of the problem, but those we are responsible for need some responsibility, of course we must look after them, and here I have a wife and two children who deserve a lot more than what I am able to offer. 
     
     - Logical consequence of the 2015 2016 Civil War. 
     
     - Logical consequence of the Russia conflict between 2017/2017. 
     
     People's economic output in the year 2078 increased by 13.5 % year-on-year, while the productivity grew by almost 20 %. GDP declined by 3.5 % during the year. Children grew by 1.3 %, their entire supply of pectin, and its dependency on the earth evaporated. An average child grew 2.9 % in the year 2078. Farmers and workers in the year 2079 produced their own food, much of it derived from farming. 
     
     - Logical consequence of the Canada-U.S. Iraq war. 
    
    

# 4. Conclusion



From my experiment, fine-tuning GPT-2 with specific task domain dataset does improve the perplexity score. Howerver, from human evaluation, I could not tell which generated story is better. The task of generative language modeling is way too hard.The human writing ablilities are far more complex than the existing technologies are able to reach. We still have a long way to explore in this field.  
