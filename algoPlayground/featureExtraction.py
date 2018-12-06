
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model,svm
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords 
import csv
import string
import pickle
import os, sys

this_directory = os.getcwd()
csv_file = open(os.path.join(this_directory,"train.csv"),"rt")

#Initialize text cleaning modules
lemma = nltk.wordnet.WordNetLemmatizer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

dataset = []
data_target = []

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

def process_data(csv_file):
	reader = csv.reader(csv_file)
	for idx,row in enumerate(reader):
		if idx == 0:
			continue
		cleaned_text = clean_text(row[2])
		rating = int(row[3])
		dataset.append(cleaned_text)
		data_target.append(rating)

process_data(csv_file)

# builds a dictionary of features and transforms dataset to feature vectors:
count_vect = CountVectorizer()
data_train = count_vect.fit_transform(dataset)

count_vect_ngram = CountVectorizer(ngram_range=(2, 2))
data_train_ngram =  count_vect_ngram.fit_transform(dataset)

# takes data from prior two lines and creates an actual dictionary and then normalize 
tf_transformer = TfidfTransformer()
data_train_tf = tf_transformer.fit_transform(data_train)

tf_transformer_ngram = TfidfTransformer()
data_train_tf_ngram = tf_transformer_ngram.fit_transform(data_train_ngram)

# pickle files
with open(os.path.join(this_directory,"tf_transformer"),'wb') as out:
    pickle.dump(tf_transformer, out)

with open(os.path.join(this_directory,"tf_transformer_ngram"),'wb') as out:
    pickle.dump(tf_transformer_ngram, out)

with open(os.path.join(this_directory,"count_vect"),'wb') as out:
    pickle.dump(count_vect, out)

with open(os.path.join(this_directory,"count_vect_ngram"),'wb') as out:
    pickle.dump(count_vect_ngram, out)

with open(os.path.join(this_directory,"data_train_tf"),'wb') as out:
    pickle.dump(data_train_tf, out)

with open(os.path.join(this_directory,"data_train_tf_ngram"),'wb') as out:
    pickle.dump(data_train_tf_ngram, out)
    
with open(os.path.join(this_directory,"data_target"),'wb') as out:
    pickle.dump(data_target, out)







