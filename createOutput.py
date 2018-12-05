import numpy as np 
import sklearn
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

from parse_emo_and_subj import score_emo, score_subj

this_directory = os.getcwd()
#csv_file = open(os.path.join(this_directory,"test.csv"),"rt")
csv_file = open(os.path.join(this_directory,'train.csv'),'rt')

#Initialize text cleaning modules
lemma = nltk.wordnet.WordNetLemmatizer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

# get all classifiers 
clf_tf = pickle.load( open( os.path.join(this_directory,'clf_tf'), "rb" ) )
clf_ngram = pickle.load( open( os.path.join(this_directory,"clf_ngram"), "rb" ) )
clf_emo = pickle.load( open( os.path.join(this_directory,"clf_emo"), "rb" ) )
clf_pos_neg = pickle( open( os.path.join(this_directory,"clf_pos_neg"), "rb" ) )

ids = []
data_tf = [] 
data_emo = [] 

# cleans the test data 
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

# this takes in the test.csv file, cleans text of each instance and puts into appropriate vector lisr
# is different from
def process_data(csv_file):
	reader = csv.reader(csv_file)
	for idx,row in enumerate(reader):
		
        if idx == 0:
			continue
		
		cleaned_text = clean_text(row[2])
		phraseId = int(row[0])
		ids.append(phraseId)
		emo_vector = score_emo(cleaned_text)
		subj_vector = score_subj(cleaned_text)

		total_vector = []
		total_vector.extend(emo_vector)
		total_vector.extend(subj_vector)
		
		data_emo.append(total_vector)
		data_tf.append(cleaned_text)

process_data(csv_file)


print(data_emo)

#count_vect = CountVectorizer()
#intput_tf  = count_vect.fit_transform(data_tf)

#tf_transformer = TfidfTransformer()
#test_tfidf = tf_transformer.fit_transform(intput_tf)

#count_vect_ngram = CountVectorizer(ngram_range=(2, 2))
#input_ngram =  count_vect_ngram.fit_transform(data_tf)

#ngram_tfidf = tf_transformer.fit_transform(input_ngram)


# create array of predicted values  
#tf = clf_tf.predict(test_tfidf.toarray())
#ngram = clf_ngram.predict(ngram_tfidf.toarray())
emo = clf_emo.predicted(data_emo)






