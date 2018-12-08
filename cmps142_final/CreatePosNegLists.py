#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 21:41:53 2018

@author: miaaltieri

this is a script that makes input into a dictionary, to create the correct
dictionaries copy and paste to words from these links:
    
http://ptrckprry.com/course/ssd/data/positive-words.txt

http://ptrckprry.com/course/ssd/data/negative-words.txt
    
"""
import pickle 
import os

positive = []
word = input()
# enter a :) to say youre done with creating this dictionary and move onto to
# the next one
while word != ':)':
    word = input()
    positive.append(word)
    
this_directory = os.getcwd()
with open(os.path.join(this_directory,"positive_words"),'wb') as out:
    pickle.dump(positive,out)
    
negative = []
word = input()
# enter a :) to say youre done with creating this dictionary 
while word != ':)':
    word = input()
    negative.append(word)
    
this_directory = os.getcwd()
with open(os.path.join(this_directory,"negative_words"),'wb') as out:
    pickle.dump(negative,out)
    