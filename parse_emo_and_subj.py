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

emo_dic = {}


def score_emo(text):
    p_file = open("Emotion-Lexicon-Dictionary.p","rb")
    emo_dic = pickle.load(p_file)
    # print(emo_dic)
    p_file.close()
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

def score_subj(text):
    return 0
# this function goes through the csv line by line and sorts calls the necessary
# functions to sort the data

if __name__ == "__main__":
    print(score_emo("foul"))
    # open file
    # csv_file = open(os.path.join(this_directory,"train.csv"),"rt")

    # # call sortData
    # sortData(csv_file)
    # print(Data_by_Emo)
    # # pickle files
    # with open(os.path.join(this_directory,"Data_by_Emo"),'wb') as out:
    #     pickle.dump(Data_by_Emo,out)
