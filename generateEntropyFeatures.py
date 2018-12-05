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

# key: rating ; value: text
text_dict = {}
COSINE_FAIL = -1







remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

#Initialize text cleaning modules
lemma = nltk.wordnet.WordNetLemmatizer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
stemmer = nltk.stem.porter.PorterStemmer()

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
    for i in range (0,5):
        entropy = cosine_sim(sentence,text_dict[i])
        entropy_vect.append(entropy)

    return entropy_vect

# this main function is used for developer purposes to see how quick the thingy
# works 
if __name__ == "__main__":
    entropy_vectors = []
    this_directory = os.getcwd()
    
    # open file
    csv_file = open(os.path.join(this_directory,"train.csv"),"rt")
    reader = csv.reader(csv_file)
        
    count = 0
    print(datetime.datetime.now())
    

    for idx,row in enumerate(reader):    
        count+=1
        if count%8 != 0:
            continue
        # skip col headers
        if idx == 0:
            continue
        
        sentence = clean_text(row[2])
        rating = int(row[3])
    
        e = score_entropy(sentence)
        
        res_row = [e,rating]
        entropy_vectors.append(res_row)
        
        if count%1010 == 0 :
            print(e, rating)
            print(count/1010,"% done")
    
    with open(os.path.join(this_directory,"entropy_vectors_complete"),'wb') as out:
        pickle.dump(entropy_vectors,out)
