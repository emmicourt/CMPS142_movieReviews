#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:48:41 2018

@author: miaaltieri
"""


"""
This is the dictionary we intend to populate in this script, the key is a rating
and the key is a list of lists, the nested lists contain 20 different LIWC ratings

"""
LIWC_dict = {}
LIWC_vectors = []
LIWC = []
flag = 0 

import nltk
import os
import pickle
import re
import datetime
nltk.download('punkt')

def score_LIWC(sentence):
    global flag
    global LIWC_dict
    if flag == 0:
        this_directory = os.getcwd()
        with open(os.path.join(this_directory,"LIWC_dict"),"rb") as f_p:
            LIWC_dict = pickle.load(f_p)
        flag = 1

    LIWC_scores = {}
    ret_LIWC = []
    
    # initalize our resulting dict
    for cat in LIWC_dict:
        LIWC_scores[cat]=0
        ret_LIWC.append(0)
        
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
        return ret_LIWC 
    
    index = 0
    # normalize the scores by dividing by sentence length 
    for key,val in LIWC_scores.items():
        if LIWC_scores[key] == 0:
           LIWC_scores[key] = .00000001
        else:
            LIWC_scores[key] = LIWC_scores[key]/sentence_length
            
        ret_LIWC[index] = LIWC_scores[key] 
        index += 1
        
    return ret_LIWC 


# this includes all LIWC category calculations in our LIWC vector
def computeLIWCvector(sentence):
    return score_LIWC(sentence.split())


if __name__ == "__main__":
    Longest_Only = None
    this_directory = os.getcwd()
    data_by_rating_dir = this_directory +'/Longest_Only'
    with open(data_by_rating_dir,"rb") as f_p:
        Longest_Only = pickle.load(f_p)
        
    with open(os.path.join(this_directory,"LIWC_dict"),"rb") as f_p:
        LIWC_dict = pickle.load(f_p)
    
    total_calcs = len(Longest_Only)
    one_percent = int(total_calcs/100)
    count = 0
    print(one_percent)
    # go through Data_by Rating, this outer loop grabs a list of all the text
    # associated with a rating
    print(datetime.datetime.now())
    for row in Longest_Only:
        rating = row[3]
        sentence = row[2]
        # LIWC ratings and appends that to our LIWC_vectors
        LIWC_vector = computeLIWCvector(sentence)
        count += 1
        if count%one_percent == 0 :
            print(LIWC_vector)
            print(count/one_percent,"% done")
            print(datetime.datetime.now())
            
        res_row = [LIWC_vector,rating]
        LIWC_vectors.append(res_row)
        
    with open(os.path.join(this_directory,"LIWC_vector"),'wb') as out:
        pickle.dump(LIWC_vectors,out)

