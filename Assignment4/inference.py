import sys
import os
import csv
import gensim
import keras
import numpy as np
import pandas as pd
import string
import pickle as pkl
from sklearn.model_selection import train_test_split
import nltk
#from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, Dropout, BatchNormalization, Activation, Bidirectional,Flatten, LSTM
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
nltk.download('stopwords')
nltk.download('punkt')

if __name__ == "__main__":
  sample = sys.argv[1] #arg1: Path to a .txt file, which contains some words compiled for evaluation.
  model_file = sys.argv[2] #path to model


  sample = [sen.rstrip('\n').translate(str.maketrans('','',string.punctuation)) for sen in open(sample)]
  word_tokens = [word_tokenize(w.lower()) for w in sample]

  for i in word_tokens:
    with open("/a4/data/tokenizer.pkl","rb") as f: #load the pickel file from data folder "tokenizer.pkl"
      xx = pkl.load(f)
    #xx = pkl.load(os.path.join('/media/aishwaryaallada/ALLADA/Aish/tokenizer.pkl','r'))
    X = xx.texts_to_sequences([' '.join(seq[:23]) for seq in i])
    Xx = pad_sequences(X, maxlen=23, padding='post', truncating='post')
    #print(Xx)
    model_1 = keras.models.load_model(os.path.join(model_file))
    # sentiment = model_file.predict(Xx,batch_size=1,verbose = 2)[0]

    # print(model_1.predict_classes(Xx))
    sentiment = model_1.predict_classes(Xx)
    # print(sentiment)

    # sentiment = (model_1.predict(Xx) > 0.5).astype("int32")
    # print(type(sentiment))

    # if 1 in sentiment:
    #   print("Pos")
    # else:
    #   print("Neg")

    # if (0 in sentiment) == True:
    #   print("negative")
    # else:
    #   print("positive")


    if(np.amax(sentiment) == 0):
      print("Negative")
    elif (np.amax(sentiment) == 1):
      print("Positive")

