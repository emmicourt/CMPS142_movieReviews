from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
import pickle
import os, sys

#starting my feature array 
features = []

#open my pickle files 
with open(os.path.join(this_directory,"Data_by_Rating"),'rb') as :
        Data_by_Rating = pickle.load(Data_by_Rating,out)
        
    with open(os.path.join(this_directory,"Data_by_Phrase"),'rb') as out:
        Data_by_Rating = pickle.load(Data_by_Phrase,out)
        
    with open(os.path.join(this_directory,"Data_by_Sentence"),'rb') as out:
        Data_by_Rating = pickle.load(Data_by_Sentence,out)
        
    with open(os.path.join(this_directory,"Stop_Word_Data"),'rb') as out:
        Data_by_Rating = pickle.load(Stop_Word_Data,out)


vec = DictVectorizer() 
vectorizer = CountVectorizer()
