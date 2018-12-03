
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

# Opend the pickle files from frequency metrics 
data_train_tf = pickle.load( open( os.path.join(this_directory,"data_train_tf"), "rb" ) )
data_target = pickle.load( open( os.path.join(this_directory,"data_target"), "rb" ) )

# Open and process emotion data into arrays 
emo_data = pickle.load( open( os.path.join(this_directory,"Data_by_Emo"), "rb" ) ).values()
emo_train_data = []
emo_target = [] 

for x in emo_data:
	emo_train_data.append(x[0:-1])
	emo_target.append(x[-1])

# randomly splitting on the data into test and train set
a_train, a_test, b_train, b_test = train_test_split(data_train_tf, data_target, test_size=0.33, random_state=42)

emo1_train, emo1_test, emo2_train, emo2_test = train_test_split(emo_train_data, emo_target, test_size=0.33, random_state=42)

## Naive Bayes 
clfTf = MultinomialNB().fit(a_train.toarray(), b_train).score(a_test, b_test)
print(clfTf)  # == 0.5730651872399445
clfEmo = MultinomialNB().fit(emo1_train, emo2_train).score(emo1_test, emo2_test)
print(clfEmo)   # == 0.5048128342245989


## KNN 
knnTf = KNeighborsClassifier(algorithm='brute').fit(a_train.toarray(), b_train).score(a_test, b_test)
print(knnTf) # == 0.5828294036061026

knnEmo = KNeighborsClassifier().fit(emo1_train, emo2_train).score(emo1_test, emo2_test)
print(knnEmo) # == 0.4894830659536542


## Logistic Regression 
log = linear_model.LogisticRegression(solver='lbfgs', max_iter= 500, multi_class='multinomial')

logTf = log.fit(a_train.toarray(), b_train).score(a_test, b_test)
print(logTf) # == 0.6151456310679612

logEmo = log.fit(emo1_train, emo2_train).score(emo1_test, emo2_test)
print(logEmo) # == 0.493048128342246

## we didnt cover this classifier 
sgd = SGDClassifier()

sgdTf = sgd.fit(a_train.toarray(), b_train).score(a_test, b_test)
print(sgdTf) # ==> 0.5687378640776699

sgdEmo = sgd.fit(emo1_train, emo2_train).score(emo1_test, emo2_test)
print(sgdEmo) # == 0.209625668449197

## This Allows us to vote with the different classifiers 
#clfVote = VotingClassifier(estimators=[('nb', clf1), ('lr', log)], 
#	voting='hard').fit(a_train.toarray(), b_train).score(a_test, b_test)
# 0.5934535367545076








