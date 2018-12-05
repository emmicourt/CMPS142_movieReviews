# import urllib
"""
this code take inputs Emotion-Lexicon-Dictionary.p and train.csv
and outputs one pickle file
    1-Data_by_Emo.pickle
    this is a dictionary where the key is the phraseId and the value is
    a 11 dimensional list with the first ten value being data for
    each emotional rating, and the last being the label
    phraseId:['anticipation', 'joy', 'negative', 'sadness', 'disgust', 'positive', 'anger', 'surprise', 'fear', 'trust', 'label']

@author: keenanyamasaki
with code from
@author: miaaltieri
"""
import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
import os, sys
import pickle
import string
from nltk.corpus import stopwords
import numpy
from numpy import sign

this_directory = os.getcwd()
# emo_dic = {}
# subj_dic = {}

def score_emo(text, emo_dic):

    #p_file = open("Emotion-Lexicon-Dictionary.p","rb")
    #emo_dic = pickle.load(p_file)

    # with open(os.path.join(this_directory,'Emotion-Lexicon-Dictionary.p'),"rb") as f_p:
    #    emo_dic = pickle.load(f_p, encoding='latin1')

    # print(emo_dic)
    # p_file.close()
    # in order 'anticipation', 'joy', 'negative', 'sadness', 'disgust',
    # 'positive', 'anger', 'surprise', 'fear', 'trust'
    phrase_data = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    num_words = 0;
    for word in text.split(' '):
        num_words += 1
        if word in emo_dic.keys():
            # print(word)
            phrase_data[0] += emo_dic[word]['anticipation']
            phrase_data[1] += emo_dic[word]['joy']
            phrase_data[2] += emo_dic[word]['negative']
            phrase_data[3] += emo_dic[word]['sadness']
            phrase_data[4] += emo_dic[word]['disgust']
            phrase_data[5] += emo_dic[word]['positive']
            phrase_data[6] += emo_dic[word]['anger']
            phrase_data[7] += emo_dic[word]['surprise']
            phrase_data[8] += emo_dic[word]['fear']
            phrase_data[9] += emo_dic[word]['trust']
        else:
            continue;
    for i in range(0,9):
        phrase_data[i] = phrase_data[i] / num_words
    # print(phrase_data)
    return phrase_data

def score_subj(text, subj_dic):
    # with open(os.path.join(this_directory,'subjective_lexicon_dic.p'),"rb") as f_p:
    #    subj_dic = pickle.load(f_p, encoding='latin1')
    
    #p_file = open("subjective_lexicon_dic.p","rb")
    #subj_dic = pickle.load(p_file)
    # print(emo_dic)
    #p_file.close()
    # num strongsubj words, number of weak subj words, pos pri words, num of neg pri words, 
    # generally more pos or neg, generally more strong or weak
    phrase_data = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    for word in text.split(' '):
        # in order: [strongsubj, weaksubj, +priorpolarity, -priorpolarity] both adds one to both + and -
        word_data = [0, 0, 0, 0]
        if word in subj_dic.keys():
            for key in subj_dic[word]:
                if subj_dic[word][key]['type'] == 'strongsubj':
                    word_data[0] += 1
                    word_data[1] -= 1
                else:
                    word_data[0] -= 1
                    word_data[1] += 1
                if subj_dic[word][key]['priorpolarity'] == 'both':
                    word_data[2] += 1
                    word_data[3] += 1
                if subj_dic[word][key]['priorpolarity'] == 'negative':
                    word_data[2] -= 1
                    word_data[3] += 1
                else:
                    word_data[2] += 1
                    word_data[3] -= 1
        else:
            continue;
        # print(word_data)
        phrase_data[4]+= word_data[0]-word_data[1]
        phrase_data[5]+= word_data[2]-word_data[3]
        for i in word_data:
            i = sign(i)
        for i in range(0,4):
            phrase_data[i] += word_data[i]
        # phrase_data[4]+= word_data[0]-word_data[1]
        # phrase_data[5]+= word_data[2]-word_data[3]
    # print(phrase_data)
    phrase_data[4] = sign(phrase_data[4])
    phrase_data[5] = sign(phrase_data[5])
    return phrase_data
# this function goes through the csv line by line and sorts calls the necessary
# functions to sort the data



