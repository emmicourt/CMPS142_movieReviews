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
data_train_tf
data_train_tf_ngram
Data_by_Emo
Emotion-Lexicon-Dictionary.p
featureExtraction.py 
models.py 
createOuput.py
parse_emo_and_subj.py
traing.csv 

featureExtraction.py: 
This file takes in the training csv file, parses and cleans the input text, and creates 
the tfidf feature vectors. The list of feature vectors are then put in the pickle files: data_train_tf, 
data_train_tf_ngram to be used by models.py

models.py: 
This files takes in the three feature vectors from the pickel files. The feature vectors are opened from the 
pickel files and processed into usable structures for sklearn. We then use the sklearn classifiers and voting
classifiers to create our models. These models are sustained using pickle files to be used by createOutput.py

createOutput.py
This file uses the models to make predictions on the test data. The three classifiers vote on a prediction
and write to the 


How to run:
===========
1. Create feature vectors and dump into pickle files: 
	python featureExtraction.py
	python 

2. Take feature vectors and train: 
	python models.py

3. Create output prediction file: 
	python createOutput.py 

