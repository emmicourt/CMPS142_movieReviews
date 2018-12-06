#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:48:41 2018

@author: miaaltieri
"""

import os
import csv
import pickle
import nltk, string
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
nltk.download('punkt')

pos_score = []
neg_score = []
senti = {}


#Initialize text cleaning modules
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
stemmer = nltk.stem.porter.PorterStemmer()
lemma = nltk.wordnet.WordNetLemmatizer()


# this cleans the text by:
#   putting everything to lowercase
#   removing punctation
#   lemmatizing
#   removing stopwords 
def clean_text (text):
    text = text.translate(remove_punctuation_map).lower()
    word_tokens = word_tokenize(text) 
    lemmatized_sentence = [lemma.lemmatize(word.lower()) 
        for word in word_tokens if word.isalpha()]
    space = ' '
    sentence = space.join(lemmatized_sentence)
    return sentence



def load_senti():
    global senti
    this_directory = os.getcwd()
    with open(os.path.join(this_directory,"Senti_dict"),"rb") as f_p:
        senti = pickle.load(f_p, encoding='latin1')

# treturns a vector or positive and negative
def score_pos_neg(sentence):
    score = 0
    
    for word in sentence:
        if word in senti:
            score += senti[word] 
    
    return score


# this main function is used for developer purposes to see how quick the thingy
# works 
if __name__ == "__main__":
    senti_vectors = []
    this_directory = os.getcwd()
    
    load_senti()
       
    # open file
    csv_file = open(os.path.join(this_directory,"train.csv"),"rt")
    reader = csv.reader(csv_file)
        
    count = 0
    print(datetime.datetime.now())
    for idx,row in enumerate(reader):    
        count+=1
        # skip col headers
        if idx == 0:
            continue
        
        sentence = clean_text(row[2])
        rating = int(row[3])

        vect = score_pos_neg(sentence.split())
        
        res_row = [vect,rating]
        pos_neg_vectors.append(res_row)
        
        if count%1010 == 0 :
            print(vect, rating)
            print(count/1010,"% done")
    
    with open(os.path.join(this_directory,"pos_neg_vectors"),'wb') as out:
        pickle.dump(pos_neg_vectors,out)
