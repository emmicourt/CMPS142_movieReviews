#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file takes in 1 csv file and produces three different python pickle files
    1 - Data_by_Rating.pickle
        this is a dictionary where the key is a rating and the value is a list
        of text
    2 - Data_by_Phrase.pickle
        this is a dictionary where the key is a phraseId in the range 500 and
        the valuse is another dictionary, this nested dictionary has a key that
        is the rating and a value of the sentences associated with it
    3 - Data_by_Sentence.pickle
        this is a dictionary where the key is a sentenceId and the valuse is
        another dictionary, this nested dictionary has a key that is the rating
        and a value of the sentences associated with it
    4 - Stop_Word_Data.pickle
        this is a dictionary where they key is a rating and the value is another
        dictionary, this nested dictionary has a key that is a stop word and a
        value that is the number of times that stop word was seen

@author: miaaltieri
"""

import csv
import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
import os, sys
import pickle
import string
from nltk.corpus import stopwords



# set up dictionaries to be pickled
Data_by_Rating = {}
Data_by_Phrase = {}
Data_by_Sentence = {}
Stop_Word_Data = {}



#Initialize text cleaning modules
lemma = nltk.wordnet.WordNetLemmatizer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)



# this cleans the text by:
#   putting everything to lowercase
#   removing punctation
#   lemmatizing
#   removing stopwords
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


# fills Data_by_Rating
def gatherRatingInfo (rating, text):
    if rating not in Data_by_Rating:
        Data_by_Rating[rating] = [text]
        return
    Data_by_Rating[rating].append(text)


# fills Data_by_Phrase
def gatherPhraseInfo (rating, phrase_id, text):
    # round to nearest 500
    rounded_phrase_id = round(phrase_id / 500.0) * 500.0

    if rounded_phrase_id not in Data_by_Phrase:
        Data_by_Phrase[rounded_phrase_id] = {}

    if rating not in Data_by_Phrase[rounded_phrase_id]:
        Data_by_Phrase[rounded_phrase_id][rating] = [text]
        return

    Data_by_Phrase[rounded_phrase_id][rating].append(text)

# fills Data_by_Sentence
def gatherSentenceInfo (rating, sentence_id, text):
    if sentence_id not in Data_by_Sentence:
        Data_by_Sentence[sentence_id] = {}

    if rating not in Data_by_Sentence[sentence_id]:
        Data_by_Sentence[sentence_id][rating] = [text]
        return

    Data_by_Sentence[sentence_id][rating].append(text)



# fills Stop_Word_Data
def gatherStopWordInfo (rating, text):
    stop_words = set(stopwords.words('english'))

    tok_text = word_tokenize(text)
    for word in tok_text:
        if word in stop_words:
            if rating not in Stop_Word_Data:
                Stop_Word_Data[rating] = {}
            if word not in Stop_Word_Data[rating]:
                Stop_Word_Data[rating][word] = 0
            Stop_Word_Data[rating][word]+=1


# this function goes through the csv line by line and sorts calls the necessary
# functions to sort the data
def sortData(csv_file):
    reader = csv.reader(csv_file)
    for idx,row in enumerate(reader):
        # skip col headers
        if idx == 0:
            continue

        sentence_id = int(row[1])
        phrase_id = int(row[0])
        cleaned_text = clean_text(row[2])
        plain_text = row[2]
        rating = int(row[3])

        gatherRatingInfo(rating,cleaned_text)
        gatherPhraseInfo(rating, phrase_id, cleaned_text)
        gatherSentenceInfo(rating, sentence_id, cleaned_text)
        gatherStopWordInfo(rating, plain_text)


if __name__ == "__main__":
    this_directory = os.getcwd()

    # open file
    csv_file = open(os.path.join(this_directory,"train.csv"),"rt")

    # call sortData
    sortData(csv_file)
    print(Stop_Word_Data)

    # pickle files
    with open(os.path.join(this_directory,"Data_by_Rating"),'wb') as out:
        pickle.dump(Data_by_Rating,out)

    with open(os.path.join(this_directory,"Data_by_Phrase"),'wb') as out:
        pickle.dump(Data_by_Phrase,out)

    with open(os.path.join(this_directory,"Data_by_Sentence"),'wb') as out:
        pickle.dump(Data_by_Sentence,out)

    with open(os.path.join(this_directory,"Stop_Word_Data"),'wb') as out:
        pickle.dump(Stop_Word_Data,out)
