#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:48:41 2018

@author: miaaltieri
"""

import os
import pickle
import nltk, string
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
nltk.download('punkt')

# key: rating ; value: text
text_dict = {}
COSINE_FAIL = -1







remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]



# this includes all LIWC category calculations in our LIWC vector
def populateTextDict():
    print('populating the dict....')
    Data_by_Rating = None
    this_directory = os.getcwd()
    data_by_rating_dir = this_directory +'/CleanData/Data_by_Rating'
    with open(data_by_rating_dir,"rb") as f_p:
        Data_by_Rating = pickle.load(f_p)
        
    for rating, text_list in Data_by_Rating.items():
        text_dict[rating]=''
        for sentence in text_list:
            text_dict[rating]+=' '+sentence

            

def score_entropy(sentence):
    if text_dict == {}:
        populateTextDict()
        
    entropy_vect = []
    for i in range (0,4):
        entropy = cosine_sim(sentence,text_dict[i])
        entropy_vect.append(entropy)

    return entropy_vect

# this main function is used for developer purposes to see how quick the thingy
# works 
if __name__ == "__main__":
    Longest_Only = None
    this_directory = os.getcwd()
    data_by_rating_dir = this_directory +'/Longest_Only'
    
    
    with open(data_by_rating_dir,"rb") as f_p:
        Longest_Only = pickle.load(f_p)
        
    sentences_to_calculate = len(Longest_Only)
    one_percent = int(sentences_to_calculate/100)
    print(one_percent)
    
    count = 0
    print(datetime.datetime.now())
    for row in Longest_Only:
        rating = row[3]
        sentence = row[2]
        e = score_entropy(sentence)
        count+=1
        if count%one_percent == 0 :
            print(e)
            print(count/one_percent,"% done")
    
