#!/usr/bin/python3

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import csv

# descriptive file in the format 'id_item,text'
dataset = 'movielens'
df = pd.read_csv(f"{dataset}/text_ml1m.csv")

ids = np.array(df["item"])
descriptions= np.array(df["description"])

# word model
sentenceTransformerName = 'all-MiniLM-L12-v2'
model = SentenceTransformer(sentenceTransformerName)

embeddings = []

# encoding sentences
for contIndex, cont in enumerate(descriptions):
  print('encoding sentence', contIndex, len(descriptions))

  try:
    embedding = model.encode(cont)
    embeddings.append(embedding)
  except:
    print('error')
    exit(1)

rows = []
with open(f'{dataset}/train.tsv') as tsvFile:
    reader = csv.reader(tsvFile, delimiter='\t')
    rows = list(reader)

usersSet = set()
for row in rows:
    userId = int(row[0])
    usersSet.add(userId)
usersCount = len(usersSet)

users = {}
for row in rows:
    userId, itemId, dis_like = [int(field) for field in row]
    if dis_like == 1:
        print(itemId, usersCount)
        users[userId] = users.get(userId, []) + [embeddings[itemId - usersCount]]

# create dictionary
dictionary = {userId: sum(users[userId]) / len(users[userId]) for userId in users} | {i + usersCount: embeddings[i] for i in range(len(ids))}

# save embeddings
pickle.dump(dictionary, open(f'{dataset}_{sentenceTransformerName}.pickle', 'wb'))
