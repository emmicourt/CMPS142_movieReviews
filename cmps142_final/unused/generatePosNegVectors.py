#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:48:41 2018

@author: miaaltieri

This script analyzes the positive and negative sentiments of an instances and
produces the appropriate vector
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

pos_words = []
neg_words = []
senti = {}


#Initialize text cleaning modules
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
stemmer = nltk.stem.porter.PorterStemmer()
lemma = nltk.wordnet.WordNetLemmatizer()


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

# loads the necessary dicitonaries to for this script
def load_pos_neg():
    global pos_words
    global neg_words
    global senti
    
    this_directory = os.getcwd()
    with open(os.path.join(this_directory,"positive_words"),"rb") as f_p:
        pos_words = pickle.load(f_p, encoding='latin1')
   
    with open(os.path.join(this_directory,"negative_words"),"rb") as f_p:
       neg_words = pickle.load(f_p, encoding='latin1')

    this_directory = os.getcwd()
    with open(os.path.join(this_directory,"Senti_dict"),"rb") as f_p:
        senti = pickle.load(f_p, encoding='latin1')

# returns a vector of positive and negative ratings along with the sentiment
def score_pos_neg(sentence):
    pos_score = 0
    neg_score = 0
    senti_score = 0
    
    for word in sentence:
        if word in pos_words:
            pos_score +=1
        if word in neg_words:
            neg_score +=1
        if word in senti:
            senti_score += float(senti[word])
            
    p_n_ratio = 0
    n_p_ratio = 0
    if neg_score == 0 and pos_score == 0:
        p_n_ratio = .0000000001
        n_p_ratio = .0000000001
    elif neg_score == 0:
        p_n_ratio = pos_score*pos_score
        n_p_ratio = .0000000001
    elif pos_score == 0:
        p_n_ratio = .0000000001
        n_p_ratio = neg_score*neg_score
    else:
        p_n_ratio = pos_score/neg_score
        n_p_ratio = neg_score/pos_score
    
    return [pos_score, neg_score, p_n_ratio, n_p_ratio, senti_score]


if __name__ == "__main__":
    pos_neg_vectors = []
    
    # load in the necessay dictionaries 
    this_directory = os.getcwd()
    load_pos_neg()

    # open file
    csv_file = open(os.path.join(this_directory,"train.csv"),"rt")
    reader = csv.reader(csv_file)
        
    for idx,row in enumerate(reader):    
        # skip col headers
        if idx == 0:
            continue
        
        sentence = clean_text(row[2])
        rating = int(row[3])

        vect = score_pos_neg(sentence.split())
        
        res_row = [vect,rating]
        pos_neg_vectors.append(res_row)
        

    with open(os.path.join(this_directory,"pos_neg_vectors"),'wb') as out:
        pickle.dump(pos_neg_vectors,out)
