#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 09:10:23 2018

@author: miaaltieri
"""
# 

import os
import pickle

parent_info = {}
this_directory = os.getcwd()
    
clf_pos_neg = pickle.load( open( os.path.join(this_directory,"clf_pos_neg"), "rb" ) )

def predict_rating(sentence):
    return -1
    
# rates a sentence based on parent info 
def score_with_respect_to_longest(this_sent, this_phrase_id,
                                  parent_phrase_id, parent_rating):
    
    feature_vect = [parent_rating, len(this_sent), parent_phrase_id-this_phrase_id]
    return feature_vect


if __name__ == "__main__":
    longest_sentence = None
    
    respect_data = []
    
    with open(os.path.join(this_directory,"Longest_Sentence"),"rb") as f_p:
        longest_sentence = pickle.load(f_p, encoding='latin1')
        
    # set up all the longest sentences
    for row in longest_sentence:
        print(row)
        sentence_id = int(row[1])
        phrase_id = int(row[0])
        text = row[2]
        
        prediction = predict_rating(sentence)
        parent_info[prediction] = [phrase_id, sentence_id, prediction]
        
    # now actually go through
    reader = csv.reader(csv_file)
    for idx,row in enumerate(reader):
        if idx == 0:
            continue
		
        phrase_id = int(row[0])
        sentence_id = int(row[1])
        cleaned_text = clean_text(row[2])
        actual_rating = int(row[3])
        
        parent_phrase_id = parent_info[sentence_id][0]
        parent_prediction = parent_info[sentence_id][3]
        
        
        respect_vect = score_with_respect_to_longest(cleaned_text, phrase_id,
                                                     parent_phrase_id, parent_prediction)
        
        respect_data.append([respect_vect,actual_rating])

        
    