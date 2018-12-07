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
LIWC_vectors = {}
LIWC = []

import nltk
import os
import pickle
from gatherLIWCfeats import score_text, default_dictionary_filename, load_dictionary
nltk.download('punkt')





# options = ['Present Tense','Personal Pronouns','Social Processes','Positive Emotion','Affective Processes','Inclusive','Achievement','Impersonal Pronouns','First Person Singular','Exclusive','Work','Second Person','Quantifiers','First Person Plural','Insight','Third Person Singular']
    
# this includes only specific category calculations into our LIWC vector
def computeSpecificLIWCvector(sentence):
    dictionary_filename = default_dictionary_filename()
    load_dictionary(dictionary_filename)
    
    scores = score_text(sentence)
    vect = []
    for cat in LIWC:
        rating = scores[cat]
        vect.append(rating)
        
    return vect

# this includes all LIWC category calculations in our LIWC vector
def computeLIWCvector(sentence):
    dictionary_filename = default_dictionary_filename()
    load_dictionary(dictionary_filename)
    return score_text(sentence)


if __name__ == "__main__":
    this_directory = os.getcwd()
    
    # open file
    Data_by_Rating = None
    data_by_rating_dir = this_directory +'/CleanedData/Data_by_Rating'
    with open(data_by_rating_dir,"rb") as f_p:
        Data_by_Rating = pickle.load(f_p)
    
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

