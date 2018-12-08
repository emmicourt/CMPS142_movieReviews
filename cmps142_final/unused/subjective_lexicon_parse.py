# import urllib
"""
this code take inputs subjective-lexicon-dic.p and train.csv
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
from numpy import sign

Data_by_Subj= {}
subj_dic = {}

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

def gather_subj_info(phraseId, text, label):
    # 
    phrase_data = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, label]
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
    phrase_data[4] = sign(phrase_data[4])
    phrase_data[5] = sign(phrase_data[5])
    Data_by_Subj[phraseId] = phrase_data

# this function goes through the csv line by line and sorts calls the necessary
# functions to sort the data
def sortData(csv_file):
    reader = csv.reader(csv_file)
    for idx,row in enumerate(reader):
        # skip col headers
        if idx == 0:
            continue

        sentence_id = int(row[0])
        phrase_id = int(row[1])
        cleaned_text = clean_text(row[2])
        plain_text = row[2]
        rating = int(row[3])
        gather_subj_info(phrase_id, cleaned_text, rating)


if __name__ == "__main__":
    this_directory = os.getcwd()

    p_file = open("subjective_lexicon_dic.p","rb")
    sub_dic = pickle.load(p_file)
    # print(sub_dic)
    p_file.close()

    # open file
    csv_file = open(os.path.join(this_directory,"train.csv"),"rt")

    # call sortData
    sortData(csv_file)
    # pickle files
    with open(os.path.join(this_directory,"Data_by_Subj"),'wb') as out:
        pickle.dump(Data_by_Subj,out)
