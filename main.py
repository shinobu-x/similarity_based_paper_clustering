import torch
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
from transformers import AutoTokenizer
device='cuda:0' if torch.cuda.is_available() else 'cpu'
embedding_size=512
tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
title=data.Field(use_vocab=False,tokenize=tokenizer.encode)
fields=[('title',title)]
dataset=data.TabularDataset(path='./data/paper_list.csv',format='csv',
                            fields=fields,skip_header=True)
loader=data.BucketIterator(dataset,batch_size=1)
for embedding in loader:
    embedding=embedding.to(device)
    embedding.title=embedding.title.view(-1,embedding.title.size(0))
    embedding.title=F.pad(input=embedding.title,
                          pad=(0,embedding_size-len(embedding)),
                          mode='constant',value=0)
