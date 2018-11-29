
import csv
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords 
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
		#row = [cleaned_text, rating]
		dataset.append(cleaned_text)
		data_target.append(rating)

process_data(csv_file)

count_vect = CountVectorizer()
data_train = count_vect.fit_transform(dataset)

tf_transformer = TfidfTransformer()
data_train_tf = tf_transformer.fit_transform(data_train)

a_train, a_test, b_train, b_test = train_test_split(data_train_tf, data_target, test_size=0.33, random_state=42)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(a_train, data_y_train)

predicted = clf.predict(data_X_test)

print(predicted)






