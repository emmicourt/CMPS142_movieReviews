
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
#data_train_tf_ngram = pickle.load( open( os.path.join(this_directory,"data_train_tf_ngram"), "rb" ) )
data_target = pickle.load( open( os.path.join(this_directory,"data_target"), "rb" ) )
#data_set = pickle.load( open( os.path.join(this_directory,"data_set"), "rb" ) )
#data_co_occur = pickle.load( open( os.path.join(this_directory,"co_occur_data"), "rb" ) )

#Longest_Only = pickle.load( open( os.path.join(this_directory,"Longest_Only"), "rb" ) )
#Longest_Only_train = []
#Longest_Only_target = []

#for x in Longest_Only: 
#	Longest_Only_train.append(x[2])
#	Longest_Only_target.append(x[-1])

#del Longest_Only_train[-1]
#del Longest_Only_target[-1]


#count_vect_ngram = CountVectorizer(ngram_range=(2, 2))
#data_train_ngram =  count_vect_ngram.fit_transform(Longest_Only_train)

#tf_transformer = TfidfTransformer()
#data_train_tf_ngram = tf_transformer.fit_transform(data_train_ngram)

# Open and process emotion data into sklearn usable arrays 
#emo_data = pickle.load( open( os.path.join(this_directory,"Data_by_Emo"), "rb" ) ).values()
#emo_train_data = []
#emo_target = [] 

#for x in emo_data:
#	emo_train_data.append(x[0:-1])
#	emo_target.append(x[-1])

# randomly splitting on the data into test and train set
a_train, a_test, b_train, b_test = train_test_split(data_train_tf, data_target, test_size=0.20, random_state=42)
#c_train, c_test, d_train, d_test = train_test_split(pickleJuice, Longest_Only_target, test_size=0.20, random_state=42)

#emo1_train, emo1_test, emo2_train, emo2_test = train_test_split(emo_train_data, emo_target, test_size=0.20, random_state=42)

## Naive Bayes 
#clfTf = MultinomialNB().fit(a_train.toarray(), b_train).score(a_test, b_test)
#clfTf_ngram = MultinomialNB().fit(c_train.toarray(), d_train).score(c_test, d_test)
#print(clfTf)  # == 0.5730651872399445 #0.5800265458373381
#print(clfTf_ngram) # == 0.5947182937434208


#clfEmo = MultinomialNB().fit(emo1_train, emo2_train).score(emo1_test, emo2_test)
#print(clfEmo)   # == 0.5048128342245989

#clfOcc = MultinomialNB().fit(c_train, d_train).score(c_test, d_test)
#print(clfOcc)


## KNN 
#knnTf = KNeighborsClassifier(algorithm='brute').fit(a_train.toarray(), b_train).score(a_test, b_test)
#print(knnTf) # == 0.5828294036061026

#knnEmo = KNeighborsClassifier().fit(emo1_train, emo2_train).score(emo1_test, emo2_test)
#print(knnEmo) # == 0.4894830659536542



## Logistic Regression 
#log = linear_model.LogisticRegression(solver='lbfgs', max_iter= 500, multi_class='multinomial')
#logTf = log.fit(a_train.toarray(), b_train).score(a_test, b_test)
#logTf_ngram	= log.fit(c_train.toarray(), d_train).score(c_test, d_test)
#print(logTf) # == 0.6151456310679612
#print(logTf_ngram)

#logEmo = log.fit(emo1_train, emo2_train).score(emo1_test, emo2_test)
#print(logEmo) # == 0.493048128342246

## This Allows us to vote with the different classifiers 
#clfVote = VotingClassifier(estimators=[('nb', clf1), ('lr', log)], 
#	voting='hard').fit(a_train.toarray(), b_train).score(a_test, b_test)
# 0.5934535367545076



# SVM mother fucker 
#### DO NOT RUN THIS SHIT IT BREAK COMPUTER 
clfSVM = svm.SVC(gamma='scale', decision_function_shape='ovo')
SVMfit = clfSVM.fit(a_train.toarray(), b_train).score(a_test, b_test)
print(SVMfit)







