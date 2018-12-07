# cmps142_project
# Emily Bettencourt enbetten@ucsc.edu 1465349 
# Mia Altieri mgaltier@ucsc.edu 1458683
# Keenan Yamasaki kyamasak@ucsc.edu 1487504

Purpose:
========
The purpose of this machine learning project is to predict the sentiment of textual phrases derived from movie reviews. Given an input phrase, our model classifies the phrase as negative (0), somewhat negative (1), 
neutral (2), some what positive (3), positive (4). 

ML/NP Libraries:
================
sklearn
numpy
nltk

Files: 
======
clf_emo
clf_tf 
clf_ngram
clf_pos_neg
count_vect
count_vect_ngram
data_target
data_train_tf
data_train_tf_ngram
Data_by_Emo
entropy_vectors
Emotion-Lexicon-Dictionary.p
featureExtraction.py 
models.py 
createOuput.py
parse_emo_and_subj.py
testset_1.csv
tf_transformer
tf_transformer_ngram
traing.csv 


featureExtraction.py: 
---------------------
Pre-req: train.csv

This file takes in the training csv file, parses and cleans the input text, and creates the tfidf feature vectors. The list of feature vectors are then put in the pickle files: data_train_tf, data_train_tf_ngram to be used by models.py. The CountVectorizer and tdifTransformer objects are sustained by the pickle files to be used by the createOutput.py. 

Output: data_target, data_train_tf, data_train_tf_ngram, tf_transformer, tf_transformer_ngram, countcount_vect, count_vect_ngram

models.py: 
----------
Pre-req: data_train_tf, data_train_tf_ngram

This files takes in the three feature vectors from the pickel files. The feature vectors are opened from the pickel files and processed into usable structures for sklearn. We then use the sklearn classifiers and voting classifiers to create our models. These models are sustained using pickle files to be used by createOutput.py

Output: clf_emo, clf_tf, clf_ngram, clf_pos_neg

createOutput.py
---------------
Pre-req: clf_emo, clf_tf, clf_ngram, clf_pos_neg, tf_transformer, tf_transformer_ngram, countcount_vect, count_vect_ngram

This file uses the models to make predictions on the test data. The three classifiers vote on a prediction and write to a output file.

Output: output.csv


How to run:
===========
1. Create feature vectors and dump into pickle files: 
	python featureExtraction.py

2. Take feature vectors and train: 
	python models.py

3. Run on test file and create output prediction file: 
	python createOutput.py 

