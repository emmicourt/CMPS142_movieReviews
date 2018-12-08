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
import csv
import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
import os, sys
import pickle
import string
from nltk.corpus import stopwords

Data_by_Emo= {}
emo_dic = {}

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

def gather_emo_info(phraseId, text, label):
    # in order 'anticipation', 'joy', 'negative', 'sadness', 'disgust',
    # 'positive', 'anger', 'surprise', 'fear', 'trust', 'label'
    phrase_data = [0,0,0,0,0,0,0,0,0,0, label]
    num_words = 0;
    #for all words incriment value if word in dictionary. 
    for word in text.split(' '):
        num_words += 1
        if word in emo_dic.keys():
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
    #for all values in phrase data other than label
    for i in range(0,10):
        phrase_data[i] = phrase_data[i] / num_words
    Data_by_Emo[phraseId] = phrase_data

# this function goes through the csv line by line and sorts calls the necessary
# functions to sort the data
def sortData(csv_file):
    reader = csv.reader(csv_file)
    for idx,row in enumerate(reader): 
        if idx == 0:
            continue

        sentence_id = int(row[0])
        phrase_id = int(row[1])
        cleaned_text = clean_text(row[2])
        plain_text = row[2]
        rating = int(row[3])
        gather_emo_info(phrase_id, cleaned_text, rating)


if __name__ == "__main__":
    this_directory = os.getcwd()
    #load dictionary
    p_file = open("Emotion-Lexicon-Dictionary.p","rb")
    emo_dic = pickle.load(p_file)
    p_file.close()

    # open file
    csv_file = open(os.path.join(this_directory,"train.csv"),"rt")

    # call sortData
    sortData(csv_file)

    # pickle files
    with open(os.path.join(this_directory,"Data_by_Emo"),'wb') as out:
        pickle.dump(Data_by_Emo,out)
