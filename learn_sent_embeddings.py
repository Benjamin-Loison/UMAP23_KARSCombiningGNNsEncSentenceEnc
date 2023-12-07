#!/usr/bin/python3

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# descriptive file in the format 'id_item,text'
dataset = 'movielens'
df = pd.read_csv(f"{dataset}/text_ml1m.csv")

ids = np.array(df["item"])
descriptions= np.array(df["description"])

# word model
sentenceTransformerName = 'all-MiniLM-L12-v1'
model = SentenceTransformer(sentenceTransformerName)

embeddings = []

# encoding sentences
for cont in descriptions:

  try:
    embedding = model.encode(cont)
    embeddings.append(embedding)
  except:
    embedding = np.zeros(len(embedding))
    embeddings.append(embedding)

# create dictionary
dictionary = {}

i = 0
while i < len(ids):
  dictionary[i] = embeddings[i]
  i += 1

# save embeddings
pickle.dump(dictionary, open(f'{dataset}_{sentenceTransformerName}.pickle', 'wb'))
