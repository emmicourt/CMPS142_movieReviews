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

liwc_dataset = pickle.load( open( os.path.join(this_directory,"LIWC_vector"), "rb" ) )

data_train_tf = pickle.load( open( os.path.join(this_directory,"data_train_tf"), "rb" ) )
data_target = pickle.load( open( os.path.join(this_directory,"data_target"), "rb" ) )

liwc_train = []
liwc_target = [] 

for x in liwc_dataset:
	liwc_train.append(x[0])
	liwc_target.append(x[1])

liwc_target[-1] = 0

# randomly splitting on the data into test and train set for LIWC 
a_train, a_test, b_train, b_test = train_test_split(liwc_train, liwc_target, test_size=0.20, random_state=42)

# randomly splitting for frequency
c_train, c_test, d_train, d_test = train_test_split(data_train_tf, data_target, test_size=0.20, random_state=42)

# NaiveBayes 
#nbclf = MultinomialNB().fit(a_train, b_train)
#print(nbclf) #0.31569664902998235

clfTf = MultinomialNB().fit(c_train.toarray(), d_train)

#Knn
#knn = KNeighborsClassifier().fit(a_train, b_train)
#print(knn) #0.25102880658436216

#Logistic Regression! 
log = linear_model.LogisticRegression(solver='lbfgs', max_iter= 500, multi_class='multinomial')
logclf = log.fit(a_train, b_train)
#print(logclf) #0.32157554379776604


#Logistic Regression but for frequency! 
logTf = log.fit(c_train.toarray(), d_train)

# voting with all three! 
clfVote = VotingClassifier(estimators=[('lr1', logclf), ('lr2', logTf), ('nb', clfTf)], voting='hard').fit(a_train, b_train).score(a_test, b_test)
print(clfVote) # 0.31393298059964725



