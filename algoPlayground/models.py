
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
data_train_tf_ngram = pickle.load( open( os.path.join(this_directory,"data_train_tf_ngram"), "rb" ) )
data_target = pickle.load( open( os.path.join(this_directory,"data_target"), "rb" ) )
#data_set = pickle.load( open( os.path.join(this_directory,"data_set"), "rb" ) )

# Open and process emotion data into sklearn usable arrays 
emo_data = pickle.load( open( os.path.join(this_directory,"Data_by_Emo"), "rb" ) ).values()
emo_train_data = []
emo_target = [] 

for x in emo_data:
	emo_train_data.append(x[0:-1])
	emo_target.append(x[-1])

# randomly splitting on the data into test and train set
tf1_train, tf1_test, tf2_train, tf2_test = train_test_split(data_train_tf, data_target, test_size=0.20, random_state=42)

ngram1_train, ngram1_test, ngram2_train, ngram2_test = train_test_split(data_train_tf_ngram, data_target, test_size=0.20, random_state=42)

emo1_train, emo1_test, emo2_train, emo2_test = train_test_split(emo_train_data, emo_target, test_size=0.20, random_state=42)


## Naive Bayes on three different inputs 
naive_bayes_Tf = MultinomialNB().fit(tf1_train.toarray(), tf2_train)
#print(naive_bayes_Tf)  #0.5800265458373381

naive_bayes_ngram = MultinomialNB().fit(ngram1_train.toarray(), ngram2_train)
#print(naive_bayes_ngram) # == 0.585289944619891

naive_bayes_Emo = MultinomialNB().fit(emo1_train, emo2_train)
#print(naive_bayes_Emo)   # == 0.5005882352941177


## KNN 
knnEmo = KNeighborsClassifier().fit(emo1_train, emo2_train)
#print(knnEmo) # == 0.4711764705882353


## Logistic Regression 
log = linear_model.LogisticRegression(solver='lbfgs', max_iter= 500, multi_class='multinomial')

logTf = log.fit(tf1_train.toarray(), tf2_train)
#.score(logTf) # == 0.6230033411140098

logEmo = log.fit(emo1_train, emo2_train)
#.score(logEmo) # == 0.4929411764705882

## This Allows us to vote with the different classifiers
est_tf = [('nbTf', naive_bayes_Tf), ('logTf', logTf)]
est_ngram = [('nbNgram', naive_bayes_ngram),]
est_emo = [('nbEmo', naive_bayes_Emo), ('knn', knnEmo), ('logEmo', logEmo)]

vote_tf = VotingClassifier(estimators=est_tf, voting='hard').fit(tf1_train.toarray(), tf2_train)
#.score(vote_tf) # 0.6003020733214335

vote_ngram = VotingClassifier(estimators=est_ngram, voting='hard').fit(ngram1_train.toarray(), ngram2_train)
#.score(vote_ngram) # 0.585289944619891

vote_emo = VotingClassifier(estimators=est_emo, voting='hard').fit(emo1_train, emo2_train)
#.score(vote_emo) # 0.4976470588235294

# pickle files
with open(os.path.join(this_directory,"clf_tf"),'wb') as out:
    pickle.dump(vote_tf, out)

with open(os.path.join(this_directory,"clf_ngram"),'wb') as out:
    pickle.dump(vote_ngram, out)

with open(os.path.join(this_directory,"clf_emo"),'wb') as out:
    pickle.dump(vote_emo, out)



