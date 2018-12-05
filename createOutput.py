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

from parse_emo_and_subj import score_emo, score_subj

from generateEntropyFeatures import score_entropy

this_directory = os.getcwd()
#csv_file = open(os.path.join(this_directory,"test.csv"),"rt")
csv_file = open(os.path.join(this_directory,"train.csv"),"rt")

# get all classifiers 
clf_tf = pickle.load( open( os.path.join(this_directory,"clf_tf"), "rb" ) )
clf_ngram = pickle.load( open( os.path.join(this_directory,"clf_ngram"), "rb" ) )
clf_emo = pickle.load( open( os.path.join(this_directory,"clf_emo"), "rb" ) )

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
		entropy_vector = score_entropy(cleaned_text)

		total_vector = []
		total_vector.extend(emo_vector)
		total_vector.extend(subj_vector)
		total_vector.extend(entropy_vector)
		
		data_emo.append(total_vector)
		dataset.append(cleaned_text)

# This votes on which label to do 
# It is weighted by order ie (x has highest weight, then y, follow)



# create array of predicted values  
tf = clf_tf.predict(dataset)
ngram = clf_ngram.predict(dataset)
emo = clf_emo.predicted(data_emo)

print(predicted_tf)

# Manually need to vote on these arrays and print to output file. 
output_csv = open('output_csv','w')
print >> output_csv, 'PhraseId,Sentiment'
for i in range(len(ids)): 
    a vote(tf[i], ngram[i], emo[i])
    print()





