#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 09:10:23 2018

@author: miaaltieri
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
    
clf_pos_neg = pickle.load( open( os.path.join(this_directory,"clf_pos_neg"), "rb" ) )

#Initialize text cleaning modules
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
stemmer = nltk.stem.porter.PorterStemmer()
lemma = nltk.wordnet.WordNetLemmatizer()


#------------------------------------------------------------------------------
def load_pos_neg():
    global pos_words
    global neg_words
    this_directory = os.getcwd()
    with open(os.path.join(this_directory,"positive_words"),"rb") as f_p:
        pos_words = pickle.load(f_p, encoding='latin1')
   
    with open(os.path.join(this_directory,"negative_words"),"rb") as f_p:
       neg_words = pickle.load(f_p, encoding='latin1')

# treturns a vector or positive and negative
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

def predict_rating(sentence):
    return -1
    
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
    
    csv_file = open(os.path.join(this_directory,"train.csv"),"rt")
    with open(os.path.join(this_directory,"Longest_Only"),"rb") as f_p:
        longest_sentence = pickle.load(f_p, encoding='latin1')
    

    # use the results from the pos neg classifier 
    for row in longest_sentence:
        cleaned_text = clean_text(row[2])
        pos_neg_vector = score_pos_neg(cleaned_text.split())
        data_pos_neg.append(pos_neg_vector)
        
        
    pos_neg_rating = clf_pos_neg.predict(data_pos_neg)
    
    # set up all the longest sentences
    for i,row in enumerate(longest_sentence):
        print(type(row[1]))
        age = input()
        sentence_id = row[1]
        phrase_id = row[0]
        
        prediction = pos_neg_rating[i]
        parent_info[prediction] = [phrase_id, sentence_id, prediction]
        
    # now actually go through
    reader = csv.reader(csv_file)
    for idx,row in enumerate(reader):
        if idx == 0:
            continue
        print("=============",row)
        phrase_id = int(row[0])
        sentence_id = int(row[1])
        cleaned_text = clean_text(row[2])
        actual_rating = int(row[3])
        
        parent_phrase_id = parent_info[sentence_id][0]
        parent_prediction = parent_info[sentence_id][2]
           
        
        respect_vect = score_with_respect_to_longest(cleaned_text, phrase_id,
                                                     parent_phrase_id, parent_prediction)
        
        if idx%1010 == 0:
            print(idx/1010,"% done")
            print(respect_vect,actual_rating)
        
        respect_data.append([respect_vect,actual_rating])
        
    with open(os.path.join(this_directory,"respect_longest_vectors"),'wb') as out:
        pickle.dump(respect_data,out)


        
    