# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 09:26:38 2017

@author: Rathod Sushma
"""

import os
import re

from bs4 import BeautifulSoup
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.cross_validation import LeaveOneOut, KFold, StratifiedKFold
from sklearn.metrics.classification import accuracy_score, confusion_matrix


import numpy as np
import pandas as pd

WORD_EMBED_SIZE = 100
SENT_EMBED_SIZE = 10
MAX_WORDS = 10
epochs=1
all_labels =['Tags']
NUM_CLASSES = 5

input_filepath = 'E:\\workspace\\mlprojects\\niki\\LabelledData.txt'

subjects = []
tags = []
for line in open(input_filepath).readlines():
    splits = line.split(' ')
    tags.append(splits[-1])
    subject = ' '.join(splits[:-1]).replace(',', '').replace('`','').replace("'","")
    subjects.append(subject)

data = {'Subject': subjects, 'Tags': tags}

df = pd.DataFrame.from_dict(data)
df.to_excel('E:\\workspace\\mlprojects\\niki\\LabelledData.xlsx', index=False)


FILEPATH = 'E:\\workspace\\mlprojects\\niki\\LabelledData.xlsx'
GLOVE_DIR = 'E:\\workspace\\softwares\\glove\\glove.6B'

class AttLayer(Layer):
    def __init__(self, **kwargs):
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="W", shape=(input_shape[-1], 1), initializer="normal") # W: (EMBED_SIZE, 1)
        self.b = self.add_weight(name="b", shape=(input_shape[1], 1), initializer="zeros") # b: (MAX_TIMESTEPS,)
        super(AttLayer, self).build(input_shape)

    def call(self, x):
        eij = K.tanh(K.dot(x, self.W) + self.b) # et: (BATCH_SIZE, MAX_TIMESTEPS)
        ai = K.softmax(eij) # at: (BATCH_SIZE, MAX_TIMESTEPS)
        oi = x * ai # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        return K.sum(oi, axis=1)        

    def compute_mask(self, input, input_mask=None):
        return None # do not pass the mask to the next layers
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    
def clean_str(review):
    review = re.sub(r"\\","",review)
    review = re.sub(r"\'", "", review)
    review = re.sub(r"\"", "", review)
    return review.strip().lower()

def get_embedding_matrix(word_index):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), errors='ignore')
    for line in f:
        try:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except:
            pass
    f.close()   
    embedding_matrix = np.random.random((len(word_index)+1, WORD_EMBED_SIZE))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


def prepare_model(word_index):
    
    embedding_matrix = get_embedding_matrix(word_index)

    sent_inputs = Input(shape=(MAX_WORDS,), dtype="float32")
    sent_emb = Embedding(input_dim=len(word_index)+1,
                         output_dim=WORD_EMBED_SIZE,
                         mask_zero=True,
                         weights=[embedding_matrix])(sent_inputs)
                         
    sent_enc = Bidirectional(GRU(SENT_EMBED_SIZE, return_sequences=True))(sent_emb)
    
    sent_att = AttLayer()(sent_enc)
    
    fc1_dropout = Dropout(0.2)(sent_att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    sent_pred = Dense(NUM_CLASSES, activation="softmax")(fc2_dropout)
    
    model = Model(inputs=sent_inputs, outputs=sent_pred)
    model.summary()
    
    return model

    
def run_model(model, X_train, Y_train, x_test, y_test):
    
    
#         print (x_test.shape)
#         print (y_test.shape)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, Y_train,  batch_size=64, epochs=epochs)
#         Ytest_ = model.predict(x_test)
#         ytest_ = np.argmax(Ytest_, axis=1)
#         ytest = np.argmax(y_test, axis=1)
#         
#         print("accuracy score: {:.3f}".format(accuracy_score(ytest, ytest_)))
#         print("\nconfusion matrix\n")
#         print(confusion_matrix(ytest, ytest_))
        scores = model.evaluate(x_test,y_test)
        return scores


def prepare_data_run_model():
#    VALIDATION_SPLIT = 0.8
    TEST_SPLIT = 0.7
    
    #### X Data Preparation ###
    orig_data = pd.read_excel(FILEPATH)
    data_reviews = orig_data['Subject']

    texts = []
    for i in range(data_reviews.shape[0]):
        text = BeautifulSoup(data_reviews[i])
        text = clean_str(text.get_text())
        texts.append(text)
    
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(texts)
    data_sequences = pad_sequences(sequences,maxlen = SENT_EMBED_SIZE)

    indices = np.arange(data_sequences.shape[0])
    np.random.shuffle(indices)
    data = data_sequences[indices]
    
    data_len = data.shape[0]
    X_train = data[0:int(TEST_SPLIT * data_len)]
    x_test = data[int(TEST_SPLIT * data_len):]
    
    labels = orig_data['Tags'].replace({'\n': ''}, regex=True)
    labels = pd.factorize(labels)[0]
    NUM_CLASSES = len(set(labels))
    labels = np_utils.to_categorical(labels, num_classes=NUM_CLASSES)
    
    y_train = labels[0:int(TEST_SPLIT * data_len)]
    y_test = labels[int(TEST_SPLIT * data_len):]
    
    ##### Prepare the model ###
    model = prepare_model(word_index)
    
    label_scores = run_model(model, X_train, y_train,  x_test, y_test)
    print ("######################")
    print (label_scores)
    print (label_scores[1]*100)
    
        
    

prepare_data_run_model()
