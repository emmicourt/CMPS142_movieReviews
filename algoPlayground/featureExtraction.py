
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

Longest_Only = pickle.load( open( os.path.join(this_directory,"Longest_Only"), "rb" ) )

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

# builds a dictionary of features and transforms dataset to feature vectors:
count_vect = CountVectorizer()
data_train = count_vect.fit_transform(dataset)

count_vect_ngram = CountVectorizer(ngram_range=(2, 2))
data_train_ngram =  count_vect_ngram.fit_transform(dataset)

# takes data from prior two lines and creates an actual dictionary and then normalize 
tf_transformer = TfidfTransformer()
data_train_tf = tf_transformer.fit_transform(data_train)
data_train_tf_ngram = tf_transformer.fit_transform(data_train_ngram)

Longest_Only_train = []
Longest_Only_target = []

for x in Longest_Only: 
    Longest_Only_train.append(x[2])
    Longest_Only_target.append(x[-1])

del Longest_Only_train[-1]
del Longest_Only_target[-1]

# Now I want to try and implement Co-occurance matrix so I need to create
## referencing this stackOverflow: https://stackoverflow.com/questions/35562789/word-word-co-occurrence-matrix 
X = count_vect.fit_transform(Longest_Only_train)
len(Longest_Only_train), len(Longest_Only_target) = X.shape
X[X > 0] = 1 # do this line first before computing cooccurrence
Xc = (X.T * X)
Xc.setdiag(0)
print(Xc.todense())

#pickle files
#with open(os.path.join(this_directory,"data_set"),'wb') as out:
#    pickle.dump(dataset, out)

#with open(os.path.join(this_directory,"data_train_tf"),'wb') as out:
#    pickle.dump(data_train_tf, out)

#with open(os.path.join(this_directory,"data_train_tf_ngram"),'wb') as out:
#    pickle.dump(data_train_tf_ngram, out)
    
#with open(os.path.join(this_directory,"data_target"),'wb') as out:
#    pickle.dump(data_target, out)


#with open(os.path.join(this_directory,"co_occur_data"),'wb') as out:
#    pickle.dump(Xc, out)






