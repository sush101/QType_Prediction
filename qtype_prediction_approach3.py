# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 09:26:38 2017

@author: Rathod Sushma
"""

import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
#from sklearn import tree

FILEPATH = 'E:\\workspace\\mlprojects\\niki\\LabelledData.txt'

subjects = []
tags = []
for line in open(FILEPATH).readlines():
    splits = line.split(' ')
    tags.append(splits[-1])
    subject = ' '.join(splits[:-1]).replace(',', '').replace('`','').replace("'","")
    subjects.append(subject)
    
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(subjects)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(subjects)
X = pad_sequences(sequences,maxlen = 20)

labels = pd.DataFrame(tags).replace({'\n': ''}, regex=True)
y = pd.factorize(labels.values.ravel())[0]

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

TEST_SPLIT = 0.7
data_len = X.shape[0]
X_train = X[0:int(TEST_SPLIT * data_len)]
X_test = X[int(TEST_SPLIT * data_len):]

y_train = y[0:int(TEST_SPLIT * data_len)]
y_test = y[int(TEST_SPLIT * data_len):]

#model = tree.DecisionTreeClassifier(criterion='gini')
#model.fit(X_train,y_train)
model = RandomForestClassifier(n_estimators = 30)
model.fit(X_train, y_train)
predicted  = model.predict(X_test)

correct = 0
for i, ytest in enumerate(y_test):
    if ytest == predicted[i]:
        correct += 1
        
print ("Accuracy is %.2f %%" % (correct/len(y_test)*100))
