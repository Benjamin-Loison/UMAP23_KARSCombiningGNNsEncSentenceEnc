#!/usr/bin/python3

import pandas as pd
import csv
import numpy as np
import tensorflow as tf
import pickle
import os
import gc
from tensorflow import keras

METRICS = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
def getOptimizer():
    return tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9)
LOSS = 'binary_crossentropy'

# this function loads data (embeddings) to be trained/test in a unique matrix X
# whose values are then fitted by the deep model
def matching_graph_bert_ids(users, items, ratings, graph_embs, word_embs):

  nu = []
  ni = []
  nr = []

  y_original = np.array(ratings)

  dim_embeddings = len(list(graph_embs.values())[0])

  dim_X_cols = 4

  def check_user_item_ids(user_id, item_id):
    return user_id in graph_embs and user_id in word_embs and item_id in graph_embs and item_id in word_embs

  # This code pattern computing actual array size then initialize it, seems to be here just for optimization.
  X_rows = 0
  for user_id, item_id in zip(users, items):

    user_id = int(user_id)
    item_id = int(item_id)

    check = check_user_item_ids(user_id, item_id)

    if check:
      X_rows += 1

  X = np.empty(shape=(X_rows,dim_X_cols,dim_embeddings))
  y = np.empty(shape=(X_rows))
  print("Loading embeddings to be fitted/tested...")

  c=0

  for i, (user_id, item_id) in enumerate(zip(users, items)):

    user_id = int(user_id)
    item_id = int(item_id)

    check = check_user_item_ids(user_id, item_id)

    if check:

      user_graph_emb = np.array(graph_embs[user_id])
      user_word_emb = np.array(word_embs[user_id])
      item_graph_emb = np.array(graph_embs[item_id])
      item_word_emb = np.array(word_embs[item_id])

      X[c][0] = user_graph_emb
      X[c][1] = item_graph_emb
      X[c][2] = user_word_emb
      X[c][3] = item_word_emb

      y[c] = y_original[i]
      
      nu.append(users[i])
      ni.append(items[i])
      nr.append(ratings[i])

      c += 1

  return X[0:c], y[0:c], dim_embeddings, nu, ni, nr


def read_ratings(filename):

  user=[]
  item=[]
  rating=[]

  with open(filename) as csv_file:

    csv_reader = csv.reader(csv_file, delimiter='\t')

    for row in csv_reader:
        user.append(int(row[0]))
        item.append(int(row[1]))
        rating.append(int(row[2]))

  return user, item, rating

def top_scores(predictions,n):

  top_n_scores_list = []
  top_n_scores = pd.DataFrame()

  for u in list(set(predictions['users'])):
    p = predictions.loc[predictions['users'] == u ]
    top_n_scores_list += [p.head(n)]

  top_n_scores = pd.DataFrame(top_n_scores_list)
  return top_n_scores


def model_entity_based(X,y,dim_embeddings,epochs,batch_size):

  model = keras.Sequential()

  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))

  x1_user = keras.layers.Dense(256, activation=tf.nn.relu)(input_users_1)
  x1_2_user = keras.layers.Dense(128, activation=tf.nn.relu)(x1_user)
  x1_3_user = keras.layers.Dense(64, activation=tf.nn.relu)(x1_2_user)

  x1_item = keras.layers.Dense(256, activation=tf.nn.relu)(input_items_1)
  x1_2_item = keras.layers.Dense(128, activation=tf.nn.relu)(x1_item)
  x1_3_item = keras.layers.Dense(64, activation=tf.nn.relu)(x1_2_item)

  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))

  x2_user = keras.layers.Dense(256, activation=tf.nn.relu)(input_users_2)
  x2_2_user = keras.layers.Dense(128, activation=tf.nn.relu)(x2_user)
  x2_3_user = keras.layers.Dense(64, activation=tf.nn.relu)(x2_2_user)

  x2_item = keras.layers.Dense(256, activation=tf.nn.relu)(input_items_2)
  x2_2_item = keras.layers.Dense(128, activation=tf.nn.relu)(x2_item)
  x2_3_item = keras.layers.Dense(64, activation=tf.nn.relu)(x2_2_item)
  
  concatenated_1 = keras.layers.Concatenate()([x1_3_user, x2_3_user])
  dense_user = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_1)
  dense_user_2 = keras.layers.Dense(32, activation=tf.nn.relu)(dense_user)
  
  concatenated_2 = keras.layers.Concatenate()([x1_3_item, x2_3_item])
  dense_item = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_2)
  dense_item_2 = keras.layers.Dense(32, activation=tf.nn.relu)(dense_item)

  concatenated = keras.layers.Concatenate()([dense_user_2, dense_item_2])
  dense = keras.layers.Dense(32, activation=tf.nn.relu)(concatenated)
  dense2 = keras.layers.Dense(16, activation=tf.nn.relu)(dense)
  dense3 = keras.layers.Dense(8, activation=tf.nn.relu)(dense2)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense3)

  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss=LOSS, optimizer=getOptimizer(), metrics=METRICS)
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=epochs, batch_size=batch_size)
  
  return model


def model_feature_based(X,y,dim_embeddings,epochs,batch_size):

  model = keras.Sequential()

  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))

  x1_user = keras.layers.Dense(256, activation=tf.nn.relu)(input_users_1)
  x1_2_user = keras.layers.Dense(128, activation=tf.nn.relu)(x1_user)
  x1_3_user = keras.layers.Dense(64, activation=tf.nn.relu)(x1_2_user)

  x1_item = keras.layers.Dense(256, activation=tf.nn.relu)(input_items_1)
  x1_2_item = keras.layers.Dense(128, activation=tf.nn.relu)(x1_item)
  x1_3_item = keras.layers.Dense(64, activation=tf.nn.relu)(x1_2_item)

  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))

  x2_user = keras.layers.Dense(256, activation=tf.nn.relu)(input_users_2)
  x2_2_user = keras.layers.Dense(128, activation=tf.nn.relu)(x2_user)
  x2_3_user = keras.layers.Dense(64, activation=tf.nn.relu)(x2_2_user)

  x2_item = keras.layers.Dense(256, activation=tf.nn.relu)(input_items_2)
  x2_2_item = keras.layers.Dense(128, activation=tf.nn.relu)(x2_item)
  x2_3_item = keras.layers.Dense(64, activation=tf.nn.relu)(x2_2_item)

  # In entity-based: concatenated_1 = keras.layers.Concatenate()([x1_3_user, x2_3_user])
  concatenated_1 = keras.layers.Concatenate()([x1_3_user, x1_3_item])
  dense_1 = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_1)
  dense_1_2 = keras.layers.Dense(32, activation=tf.nn.relu)(dense_1)

  # In entity-based: concatenated_2 = keras.layers.Concatenate()([x1_3_item, x2_3_item])
  concatenated_2 = keras.layers.Concatenate()([x2_3_user, x2_3_item])
  dense_2 = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_2)
  dense_2_2 = keras.layers.Dense(32, activation=tf.nn.relu)(dense_2)

  concatenated = keras.layers.Concatenate()([dense_1_2, dense_2_2])
  dense = keras.layers.Dense(32, activation=tf.nn.relu)(concatenated)
  dense2 = keras.layers.Dense(16, activation=tf.nn.relu)(dense)
  dense3 = keras.layers.Dense(8, activation=tf.nn.relu)(dense2)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense3)

  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss=LOSS, optimizer=getOptimizer(), metrics=METRICS)
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=epochs, batch_size=batch_size)

  return model

def model_single_feature_based(X,y,dim_embeddings,epochs,batch_size, feature_offset):

  model = keras.Sequential()

  input_users = keras.layers.Input(shape=(dim_embeddings,))
  input_items = keras.layers.Input(shape=(dim_embeddings,))

  x_user = keras.layers.Dense(256, activation=tf.nn.relu)(input_users)
  x_2_user = keras.layers.Dense(128, activation=tf.nn.relu)(x_user)
  x_3_user = keras.layers.Dense(64, activation=tf.nn.relu)(x_2_user)

  x_item = keras.layers.Dense(256, activation=tf.nn.relu)(input_items)
  x_2_item = keras.layers.Dense(128, activation=tf.nn.relu)(x_item)
  x_3_item = keras.layers.Dense(64, activation=tf.nn.relu)(x_2_item)

  concatenated = keras.layers.Concatenate()([x_3_user, x_3_item])
  dense = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated)
  dense_2 = keras.layers.Dense(32, activation=tf.nn.relu)(dense)
  dense_3 = keras.layers.Dense(16, activation=tf.nn.relu)(dense_2)
  dense_4 = keras.layers.Dense(8, activation=tf.nn.relu)(dense_3)

  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense_4)

  model = keras.models.Model(inputs=[input_users,input_items],outputs=out)
  model.compile(loss=LOSS, optimizer=getOptimizer(), metrics=METRICS)
  model.fit([X[:,feature_offset + 0],X[:,feature_offset + 1]], y, epochs=epochs, batch_size=batch_size)

  return model


def model_entity_based_together_is_better(X,y,dim_embeddings,epochs,batch_size):

  model = keras.Sequential()

  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))

  x1_user = keras.layers.Dense(64, activation=tf.nn.relu)(input_users_1)

  x1_item = keras.layers.Dense(64, activation=tf.nn.relu)(input_items_1)

  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))

  x2_user = keras.layers.Dense(64, activation=tf.nn.relu)(input_users_2)

  x2_item = keras.layers.Dense(64, activation=tf.nn.relu)(input_items_2)

  concatenated_1 = keras.layers.Concatenate()([x1_user, x2_user])
  dense_user = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_1)

  concatenated_2 = keras.layers.Concatenate()([x1_item, x2_item])
  dense_item = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_2)

  concatenated = keras.layers.Concatenate()([dense_user, dense_item])
  dense = keras.layers.Dense(32, activation=tf.nn.relu)(concatenated)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense)

  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss=LOSS, optimizer=getOptimizer(), metrics=METRICS)
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=epochs, batch_size=batch_size)

  return model


def model_feature_based_together_is_better(X,y,dim_embeddings,epochs,batch_size):

  model = keras.Sequential()

  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))

  x1_user = keras.layers.Dense(64, activation=tf.nn.relu)(input_users_1)

  x1_item = keras.layers.Dense(64, activation=tf.nn.relu)(input_items_1)

  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))

  x2_user = keras.layers.Dense(64, activation=tf.nn.relu)(input_users_2)

  x2_item = keras.layers.Dense(64, activation=tf.nn.relu)(input_items_2)

  concatenated_1 = keras.layers.Concatenate()([x1_user, x1_item])
  dense_1 = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_1)

  concatenated_2 = keras.layers.Concatenate()([x2_user, x2_item])
  dense_2 = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_2)

  concatenated = keras.layers.Concatenate()([dense_1, dense_2])
  dense = keras.layers.Dense(32, activation=tf.nn.relu)(concatenated)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense)

  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss=LOSS, optimizer=getOptimizer(), metrics=METRICS)
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=epochs, batch_size=batch_size)

  return model


def model_single_feature_based_together_is_better(X,y,dim_embeddings,epochs,batch_size, feature_offset):

  model = keras.Sequential()

  input_users = keras.layers.Input(shape=(dim_embeddings,))
  input_items = keras.layers.Input(shape=(dim_embeddings,))

  # They do not precise whether or not it is a ReLU in these layers.
  dense_layer_1 = keras.layers.Dense(64, activation=tf.nn.relu)(input_users)

  dense_layer_2 = keras.layers.Dense(64, activation=tf.nn.relu)(input_items)

  concatenated = keras.layers.Concatenate()([dense_layer_1, dense_layer_2])
  dense_layer_3 = keras.layers.Dense(32, activation=tf.nn.relu)(concatenated)

  dense_layer_4 = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense_layer_3)

  model = keras.models.Model(inputs=[input_users,input_items],outputs=dense_layer_4)
  model.compile(loss=LOSS, optimizer=getOptimizer(), metrics=METRICS)
  model.fit([X[:,feature_offset + 0],X[:,feature_offset + 1]], y, epochs=epochs, batch_size=batch_size)

  return model


def model_entity_dropout_selfatt_crossatt(X,y,dim_embeddings,epochs,batch_size, value):

  model = keras.Sequential()

  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))

  x1_user_drop = keras.layers.Dropout(value, input_shape=(dim_embeddings,))(input_users_1)
  x1_item_drop = keras.layers.Dropout(value, input_shape=(dim_embeddings,))(input_items_1)

  x1_user = keras.layers.Dense(512, activation=tf.nn.relu)(x1_user_drop)
  x1_2_user = keras.layers.Dense(256, activation=tf.nn.relu)(x1_user)
  x1_3_user = keras.layers.Dense(128, activation=tf.nn.relu)(x1_2_user)

  x1_item = keras.layers.Dense(512, activation=tf.nn.relu)(x1_item_drop)
  x1_2_item = keras.layers.Dense(256, activation=tf.nn.relu)(x1_item)
  x1_3_item = keras.layers.Dense(128, activation=tf.nn.relu)(x1_2_item)

  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))

  x2_user_drop = keras.layers.Dropout(value, input_shape=(dim_embeddings,))(input_users_2)
  x2_item_drop = keras.layers.Dropout(value, input_shape=(dim_embeddings,))(input_items_2)

  x2_user = keras.layers.Dense(512, activation=tf.nn.relu)(x2_user_drop)
  x2_2_user = keras.layers.Dense(256, activation=tf.nn.relu)(x2_user)
  x2_3_user = keras.layers.Dense(128, activation=tf.nn.relu)(x2_2_user)

  x2_item = keras.layers.Dense(512, activation=tf.nn.relu)(x2_item_drop)
  x2_2_item = keras.layers.Dense(256, activation=tf.nn.relu)(x2_item)
  x2_3_item = keras.layers.Dense(128, activation=tf.nn.relu)(x2_2_item)

  # self attention 1 - merge graph user and word user
  concat_user = keras.layers.Concatenate()([x1_3_user, x2_3_user])
  attention_w_user = keras.layers.Dense(128, activation='softmax')(concat_user)
  merged_user = attention_w_user * x1_3_user + (1 - attention_w_user) * x2_3_user

  # self attention 2 - merge graph item and word item
  concat_item = keras.layers.Concatenate()([x1_3_item, x2_3_item])
  attention_w_item = keras.layers.Dense(128, activation='softmax')(concat_item)
  merged_item = attention_w_item * x1_3_item + (1 - attention_w_item) * x2_3_item

  # cross attention - merge of both merged
  attention_weights = keras.layers.Dot(axes=-1)([merged_user, merged_item])
  attention_weights = keras.layers.Dense(128, activation='softmax')(attention_weights)
  merged = keras.layers.Add()([merged_user * attention_weights, merged_item * (1 - attention_weights)])

  merged2 = keras.layers.Dense(64, activation=tf.nn.relu)(merged)
  merged3 = keras.layers.Dense(32, activation=tf.nn.relu)(merged2)
  merged4 = keras.layers.Dense(16, activation=tf.nn.relu)(merged3)
  merged5 = keras.layers.Dense(8, activation=tf.nn.relu)(merged4)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(merged5)

  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss=LOSS, optimizer=getOptimizer(), metrics=METRICS)
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=epochs, batch_size=batch_size)
  
  return model

# for this implementation, graph embeddings and word embeddings 
# are dict in the form:
# id -> embedding
# where id is an integer
# embedding is a list of float or numpy.array

dataset = 'movielens'
source_graph_path = f'{dataset}/{dataset}_CompGCN_k=384.pickle'
source_text_path = f'{dataset}/{dataset}_all-MiniLM-L12-v2.pickle'
model_path = f'{dataset}/model.h5'
predictions_path = f'{dataset}/predictions'


# read training data
users, items, ratings = read_ratings(f'{dataset}/train.tsv')

# read graph and word embedding
graph_emb = pickle.load(open(source_graph_path, 'rb'))
word_emb = pickle.load(open(source_text_path, 'rb'))


# if the model already exists, it's loaded
if os.path.exists(model_path):

  recsys_model = tf.keras.models.load_model(model_path)
  print("Model loaded.")

# otherwise it's trained
else:

  print("Matched data for training...")
  X, y, dim_embeddings, _, _, _ = matching_graph_bert_ids(users, items, ratings, graph_emb, word_emb)
  
  # training the model
  epochs = 25
  batch_size = 512
  feature_offset = 0#2
  #recsys_model = model_single_feature_based_together_is_better(X,y,dim_embeddings,epochs,batch_size,feature_offset)
  loopIndex = 0
  def runModel(loopIndex, modelName, modelFunction, modelFunctionArguments):
      gc.collect()
      print(f'{loopIndex} {modelName}')
      modelFunction(*modelFunctionArguments)

  modelFunctionArguments = [X,y,dim_embeddings,epochs,batch_size]
  runModels = [#[[f'single_feature {i}', model_single_feature_based_together_is_better, modelFunctionArguments + [i]] for i in [0, 2]] + [
    ['single_feature 0', model_single_feature_based_together_is_better, modelFunctionArguments + [0]],
    ['single_feature 2', model_single_feature_based_together_is_better, modelFunctionArguments + [2]],
    ['feature based', model_feature_based_together_is_better, modelFunctionArguments],
    ['entity based', model_entity_based_together_is_better, modelFunctionArguments],
  ]
  # Assume that `model.fit` does not change its arguments.
  while True:
    for modelName, modelFunction, modelArguments in runModels:
        runModel(loopIndex, modelName, modelFunction, modelArguments)
    loopIndex += 1
  recsys_model = model_feature_based_together_is_better(*modelFunctionArguments)

  # saving the model
  recsys_model.save(model_path)

# read test ratings to be predicted
users, items, ratings = read_ratings(f'{dataset}/test.tsv')

# embeddings for test
X, y, dim_embeddings, nu, ni, nr = matching_graph_bert_ids(users, items, ratings, graph_emb, word_emb)

# predict   
print("\tPredicting...")
score = recsys_model.predict([X[:,0],X[:,1],X[:,2],X[:,3]])

# write predictions
print("\tComputing predictions...")
score = score.reshape(1, -1)[0,:]
predictions = pd.DataFrame()
predictions['users'] = np.array(nu)
predictions['items'] = np.array(ni)
predictions['scores'] = score

predictions = predictions.sort_values(by=['users', 'scores'],ascending=[True, False])

# create predictions folder if it does not exist
if not os.path.exists(predictions_path):
  os.mkdir(predictions_path)

# write top 5 predictions
top_5_scores = top_scores(predictions,5)
top_5_scores.to_csv(predictions_path + '/top5_predictions.tsv',sep='\t',header=False,index=False)
print("\tTop 5 wrote.")
