#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:48:41 2018

@author: miaaltieri
"""


"""
This is the dictionary we intend to populate in this script, the key is a rating
and the key is a list of lists, the nested lists contain 10 different LIWC ratings

i.e LIWC_vectors = 
    {
     1: [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9 10,10 ], ..., ..., ... ]
     2: []
     3: []
     4: []
     5: []
    }
"""
LIWC_dict = {}
LIWC_vectors = {}
LIWC = []

import nltk
import os
import pickle
import re
nltk.download('punkt')



def score_text(sentence):
    LIWC_scores = {}
    
    # initalize our resulting dict
    for cat in LIWC_dict:
        LIWC_scores[cat]=0
        
    # go through each word in the sentece, and for each word see if it relates
    # to a LIWC category
    # go through our LIWC cateogories, and for each word in each category see
    # if there is a match
    for word in sentence:
        for cat,vals in LIWC_dict.items():
            for regex in vals:
                if regex[-1] == '*':
                    pattern = re.compile(regex) 
                    if pattern.match(word):
                        LIWC_scores[cat] += 1 
                else:
                    if word == regex:
                        LIWC_scores[cat] += 1 

    sentence_length = len(sentence)           
    if sentence_length == 0:
        return
    # normalize the scores by dividing by sentence length 
    for key,val in LIWC_scores.items():
        LIWC_scores[key] = LIWC_scores[key]/sentence_length
                    
    # regularize scores

# this includes only specific category calculations into our LIWC vector
def computeSpecificLIWCvector(sentence):
    scores = score_text(sentence)
    vect = []
    for cat in LIWC:
        rating = scores[cat]
        vect.append(rating)
        
    return vect

# this includes all LIWC category calculations in our LIWC vector
def computeLIWCvector(sentence):
    return score_text(sentence.split())


if __name__ == "__main__":
    Data_by_Rating = None
    this_directory = os.getcwd()
    data_by_rating_dir = this_directory +'/CleanData/Data_by_Rating'
    with open(data_by_rating_dir,"rb") as f_p:
        Data_by_Rating = pickle.load(f_p)
        
    with open(os.path.join(this_directory,"LIWC_dict"),"rb") as f_p:
        LIWC_dict = pickle.load(f_p)
    
    # go through Data_by Rating, this outer loop grabs a list of all the text
    # associated with a rating
    for rating, text_list in Data_by_Rating.items():
        # this inner loop grabs a sentence associate with a rating, then computes
        # LIWC ratings and appends that to our LIWC_vectors
        for sentence in text_list:
            LIWC_vector = computeLIWCvector(sentence)
            if rating not in LIWC_vectors:
                LIWC_vectors[rating]= []
            LIWC_vectors[rating].append(LIWC_vector)
            
        
    with open(os.path.join(this_directory,"LIWC_vector"),'wb') as out:
        pickle.dump(LIWC_vector,out)

