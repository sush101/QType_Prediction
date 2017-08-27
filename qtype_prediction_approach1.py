# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 18:15:04 2017

@author: Rathod Sushma
"""


filepath = 'E:\\workspace\\mlprojects\\niki\\LabelledData.txt'

subjects = []
tags = []
for line in open(filepath).readlines():
    splits = line.split(' ')
    tags.append(splits[-1])
    subject = ' '.join(splits[:-1]).replace(',', '').replace('`','').replace("'","")
    subjects.append(subject)
    
data = {'Subject': subjects, 'Tags': tags}

#import pandas as pd
#df = pd.DataFrame.from_dict(data)
#df.to_excel('C:\\mydrive\\mlprojects\\niki\\LabelledData.xlsx', index=False)


correct =0
incorrect = 0
for i, sub in enumerate(subjects):
    result = ''
    tag = tags[i].strip()
    if sub.startswith('what') and (sub.startswith('time', 5) or sub.startswith('year', 5)):
        result = 'when'
    elif sub.startswith('what') or ('in what' in sub): 
        result = 'what'
    elif sub.startswith('when'):
        result = 'when'
    elif sub.startswith('who'):
        result = 'who'
    elif (sub.startswith('is') or
          sub.startswith('can') or 
          sub.startswith('do') or 
          sub.startswith('could') or 
          sub.startswith('would') or 
          sub.startswith('will') or 
          sub.startswith('are') or 
          sub.startswith('has') 
          ):
        result = 'affirmation'
    else:
        result = 'unknown'
        
    if result == tag:
        correct += 1
    else:
        incorrect += 1
    
    if result != tag:
        print (sub, tag, result)
    
print ("Total Correct : ",correct)
print ("Total incorrect: ", incorrect)

print ("Accuracy: %.2f%% " % (100*correct/len(subjects)))
