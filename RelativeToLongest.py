#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 09:10:23 2018

@author: miaaltieri

This code generates a feature vecotr that is based off of an instances parent 
sentence

"""
# 

import os
import pickle
import csv
import nltk, string
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
nltk.download('punkt')

parent_info = {}
this_directory = os.getcwd()
    
# loading in our pos neg classifer 
clf_pos_neg = pickle.load( open( os.path.join(this_directory,"clf_pos_neg"), "rb" ) )

#Initialize text cleaning modules
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
stemmer = nltk.stem.porter.PorterStemmer()
lemma = nltk.wordnet.WordNetLemmatizer()

# Code for posive/negative feature extraction
#------------------------------------------------------------------------------
# loading in pos/neg dictionaries
def load_pos_neg():
    global pos_words
    global neg_words
    this_directory = os.getcwd()
    with open(os.path.join(this_directory,"positive_words"),"rb") as f_p:
        pos_words = pickle.load(f_p, encoding='latin1')
   
    with open(os.path.join(this_directory,"negative_words"),"rb") as f_p:
       neg_words = pickle.load(f_p, encoding='latin1')

# returns a vector or positive and negative
def score_pos_neg(sentence):
    pos_score = 0
    neg_score = 0
    
    for word in sentence:
        if word in pos_words:
            pos_score +=1
        if word in neg_words:
            neg_score +=1
            
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
    
    return [pos_score, neg_score, p_n_ratio, n_p_ratio]
#------------------------------------------------------------------------------

# cleans the test data 
def clean_text (text):
    text = text.translate(remove_punctuation_map).lower()
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [lemma.lemmatize(word.lower()) 
        for word in filtered_sentence if word.isalpha()]
    space = ' '
    sentence = space.join(filtered_sentence)
    return sentence

    
# rates a sentence based on parent info 
def score_with_respect_to_longest(this_sent, this_phrase_id,
                                  parent_phrase_id, parent_rating):
    
    feature_vect = [parent_rating, len(this_sent), parent_phrase_id-this_phrase_id]
    return feature_vect


if __name__ == "__main__":
    longest_sentence = None
    load_pos_neg()
    data_pos_neg = []
    respect_data = []
    
    # opens necessay files
    csv_file = open(os.path.join(this_directory,"train.csv"),"rt")
    with open(os.path.join(this_directory,"Longest_Only"),"rb") as f_p:
        longest_sentence = pickle.load(f_p, encoding='latin1')
    

    # use the results from the pos neg classifier 
    for row in longest_sentence.tolist():
        cleaned_text = clean_text(row[2])
        pos_neg_vector = score_pos_neg(cleaned_text.split())
        data_pos_neg.append(pos_neg_vector)
        
    # generates predictions
    pos_neg_rating = clf_pos_neg.predict(data_pos_neg)
    
    # set up all the necessary data for longest sentences
    for i,row in enumerate(longest_sentence.tolist()):
        try:
            sentence_id = int(row[1])
        except:
            continue
        sentence_id = int(row[1])
        phrase_id = int(row[0])
        prediction = pos_neg_rating[i]
        parent_info[sentence_id] = [phrase_id, sentence_id, prediction]
        
    # now actually go through our instances and create our feature vectors
    reader = csv.reader(csv_file)
    for idx,row in enumerate(reader):
        if idx == 0:
            continue
        phrase_id = int(row[0])
        sentence_id = int(row[1])
        cleaned_text = clean_text(row[2])
        actual_rating = int(row[3])
        
        parent_phrase_id = parent_info[sentence_id][0]
        parent_prediction = parent_info[sentence_id][2]
           
        
        respect_vect = score_with_respect_to_longest(cleaned_text, phrase_id,
                                                     parent_phrase_id, parent_prediction)
            
        respect_data.append([respect_vect,actual_rating])
        
    # dumping our feature vector
    with open(os.path.join(this_directory,"respect_longest_vectors"),'wb') as out:
        pickle.dump(respect_data,out)


        
    