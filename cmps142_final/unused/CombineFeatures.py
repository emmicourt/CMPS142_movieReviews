#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:51:25 2018

@author: miaaltieri
"""

import csv
import nltk
import string
import os
import pickle
import datetime
nltk.download('punkt')
from nltk import word_tokenize

from  parse_emo_and_subj import score_emo, score_subj
from generateLIWCfeatures import score_LIWC

#Initialize text cleaning modules
lemma = nltk.wordnet.WordNetLemmatizer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
stemmer = nltk.stem.porter.PorterStemmer()

# this cleans the text by:
#   putting everything to lowercase
#   removing punctation
#   lemmatizing
def clean_text (text):
    text = text.translate(remove_punctuation_map).lower()
    word_tokens = word_tokenize(text) 
    lemmatized_sentence = [lemma.lemmatize(word.lower()) 
        for word in word_tokens if word.isalpha()]
    space = ' '
    sentence = space.join(lemmatized_sentence)
    return sentence

if __name__ == "__main__":
    features = [] 
    
    # opening correct files 
    this_directory = os.getcwd()
    csv_file = open(os.path.join(this_directory,"train.csv"),"rt")
        
    with open(os.path.join(this_directory,"LIWC_dict"),"rb") as f_p:
        LIWC_dict = pickle.load(f_p)
    
    # open file
    csv_file = open(os.path.join(this_directory,"train.csv"),"rt")
    
    reader = csv.reader(csv_file)
    for idx,row in enumerate(reader):    
        # skip col headers
        if idx == 0:
            continue
        
        sentence = clean_text(row[2])
        rating = int(row[3])
    
        # LIWC ratings and appends that to our LIWC_vectors
        LIWC_vector = score_LIWC(sentence.split())
        emo_vector = score_emo(sentence)
        subj_vector = score_subj(sentence)
        
        # adding these to our enitre feature vector
        total_vector = []
        total_vector.extend(LIWC_vector)
        total_vector.extend(emo_vector)
        total_vector.extend(subj_vector)
        
        res_row = [total_vector,rating]
        features.append(res_row)
        
    # dumping the output
    with open(os.path.join(this_directory,"LIWC_vector"),'wb') as out:
        pickle.dump(features,out)