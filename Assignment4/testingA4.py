import os
import sys
import csv
import gensim
import keras
import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, Dropout, BatchNormalization, Activation, Bidirectional,Flatten, LSTM
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from tensorflow.keras import regularizers
nltk.download('stopwords')
nltk.download('punkt')

def test(data_dir):    

  # w2v = Word2Vec.load("C:\\Users\\aisha\\OneDrive\\Desktop\\MSCI 641\\Assignments\\20816326_akallada_a3\\data\\w2v.model")
  w2v = Word2Vec.load(os.path.join(data_dir, 'w2v.model'))
  embedding_matrix = w2v.wv.vectors
  embedding_layer = Embedding(input_dim = embedding_matrix.shape[0], output_dim = embedding_matrix.shape[1],weights=[embedding_matrix],trainable=True,input_length=23)

  with open(os.path.join(data_dir, 'train_withStopWords.csv')) as f:
  	  file = csv.reader(f)
  	  train_WithSW = list(file)

  with open(os.path.join(data_dir, 'val_withStopWords.csv')) as f:
    file = csv.reader(f)
    val_WithSW = list(file)

  with open(os.path.join(data_dir, 'test_withStopWords.csv')) as f:
    file = csv.reader(f)
    test_WithSW = list(file)


  tokenizer = Tokenizer(num_words = embedding_matrix.shape[0])
  tokenizer.fit_on_texts([' '.join(seq[:23]) for seq in train_WithSW])
  pkl.dump(tokenizer,open("a4/data/tokenizer.pkl","wb")) #saving the tokenizer in pickel file for the usage in inference.py file
  X = tokenizer.texts_to_sequences([' '.join(seq[:23]) for seq in train_WithSW])
  X = pad_sequences(X, maxlen=23, padding='post', truncating='post')

  YY = tokenizer.texts_to_sequences([' '.join(seq[:23]) for seq in val_WithSW])
  YY = pad_sequences(YY, maxlen=23, padding='post', truncating='post')

  Z = tokenizer.texts_to_sequences([' '.join(seq[:23]) for seq in test_WithSW])
  Z = pad_sequences(Z, maxlen=23, padding='post', truncating='post')

  labs = []
  with open(os.path.join(data_dir, 'labels.csv')) as f:
    for line in f:
      labs.append(line)

  labelss = labs[: len(labs) - 1] 
  labelss = [x.replace('\n', '') for x in labelss]

  for n, i in enumerate(labelss):
    if i == "Positive":
      labelss[n] = 1
    else:
      labelss[n] = 0

  train_labels = labelss[:640000]
  test_labels = labelss[640000:720000]
  val_labels = labelss[720000:]

  y_train = np_utils.to_categorical(train_labels)
  y_test = np_utils.to_categorical(test_labels)
  y_val = np_utils.to_categorical(val_labels)

  EMBEDDING_DIM = 300
  BATCH_SIZE = 512
  N_EPOCHS = 10 
   
  model = Sequential()
  model.add(Embedding(input_dim = embedding_matrix.shape[0], output_dim = embedding_matrix.shape[1],weights=[embedding_matrix],trainable=True,input_length=23))
  model.add(LSTM(128,activation="relu",bias_regularizer=regularizers.l2(1e-2)))
  model.add(Dropout(rate=0.2, name='dropout_1'))
  model.add(Dense(2, activation='softmax', name='output_layer'))
  model.summary()

  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  model.fit(X, y_train,
             batch_size=BATCH_SIZE,
            epochs=10,
            validation_data=(YY, y_val))
  model.save(os.path.join(data_dir, 'nn_relu'))
  #model = keras.models.load_model(os.path.join(data_dir, 'nn_relu.model'))
  score = model.evaluate(Z, y_test, batch_size=BATCH_SIZE)
  print("ReLU Activation Function",score)


  model_1 = Sequential()
  model_1.add(Embedding(input_dim = embedding_matrix.shape[0], output_dim = embedding_matrix.shape[1],weights=[embedding_matrix],trainable=True,input_length=23))
  model_1.add(LSTM(128,activation="sigmoid",bias_regularizer=regularizers.l2(1e-2)))
  model_1.add(Dropout(rate=0.2, name='dropout_1'))
  model_1.add(Dense(2, activation='softmax', name='output_layer'))
  model_1.summary()
  model_1.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  model_1.fit(X, y_train,
             batch_size=BATCH_SIZE,
            epochs=10,
            validation_data=(YY, y_val))
  model_1.save(os.path.join(data_dir, 'nn_sigmoid'))
  #model_1 = keras.models.load_model(os.path.join(data_dir, 'nn_sigmoid.model'))
  score_1 = model_1.evaluate(Z, y_test, batch_size=BATCH_SIZE)
  print("Sigmoid Activation Function",score_1)


  model_2 = Sequential()
  model_2.add(Embedding(input_dim = embedding_matrix.shape[0], output_dim = embedding_matrix.shape[1],weights=[embedding_matrix],trainable=True,input_length=23))
  model_2.add(LSTM(128,activation="tanh",bias_regularizer=regularizers.l2(1e-2)))
  model_2.add(Dropout(rate=0.2, name='dropout_1'))
  model_2.add(Dense(2, activation='softmax', name='output_layer'))
  model_2.summary()

  model_2.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  model_2.fit(X, y_train,
             batch_size=BATCH_SIZE,
            epochs=10,
            validation_data=(YY, y_val))
  model_2.save(os.path.join(data_dir, 'nn_tanh'))
  #model_2 = keras.models.load_model(os.path.join(data_dir, 'nn_tanh.model'))
  score_2 = model_2.evaluate(Z, y_test, batch_size=BATCH_SIZE)
  print("Tanh Activation Function",score_2)
