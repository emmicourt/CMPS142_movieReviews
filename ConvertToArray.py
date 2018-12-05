#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 21:41:53 2018

@author: miaaltieri
"""
import pickle 
import os

positive = []
word = input()
while word != ':)':
    word = input()
    positive.append(word)
    
print('done with pos')
    
this_directory = os.getcwd()
with open(os.path.join(this_directory,"positive_words"),'wb') as out:
    pickle.dump(positive,out)
    
negative = []
word = input()
while word != ':)':
    word = input()
    negative.append(word)
    
this_directory = os.getcwd()
with open(os.path.join(this_directory,"negative_words"),'wb') as out:
    pickle.dump(negative,out)
    