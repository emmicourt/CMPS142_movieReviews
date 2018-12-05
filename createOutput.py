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

this_directory = os.getcwd()
#csv_file = open(os.path.join(this_directory,"test.csv"),"rt")
csv_file = open(os.path.join(this_directory,"train.csv"),"rt")


#Initialize text cleaning modules
lemma = nltk.wordnet.WordNetLemmatizer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

# get all classifiers 
clf_tf = pickle.load( open( os.path.join(this_directory,"clf_tf"), "rb" ) )
clf_ngram = pickle.load( open( os.path.join(this_directory,"clf_ngram"), "rb" ) )
clf_emo = pickle.load( open( os.path.join(this_directory,"clf_emo"), "rb" ) )
clf_pos_neg = pickle.load( open( os.path.join(this_directory,"clf_pos_neg"), "rb" ) )

ids = []
data_tf = [] 
data_emo = [] 
data_pos_neg = []


#-----------------------------------------------------------------------------
def load_pos_neg():
    global pos_words
    global neg_words
    this_directory = os.getcwd()
    with open(os.path.join(this_directory,"positive_words"),"rb") as f_p:
        pos_words = pickle.load(f_p, encoding='latin1')
   
    with open(os.path.join(this_directory,"negative_words"),"rb") as f_p:
       neg_words = pickle.load(f_p, encoding='latin1')

# treturns a vector or positive and negative
def score_pos_neg(sentence):
    pos_score = 0
    neg_score = 0
    
    for word in sentence:
        if word in pos_words:
            pos_score +=1
        if word in neg_words:
            neg_score +=1
            
    p_n_ratio = 0
    n_p_ratio = 0
    if neg_score == 0 and pos_score == 0:
        p_n_ratio = .0000000001
        n_p_ratio = .0000000001
    elif neg_score == 0:
        p_n_ratio = pos_score*pos_score
        n_p_ratio = .0000000001
    elif pos_score == 0:
        p_n_ratio = .0000000001
        n_p_ratio = neg_score*neg_score
    else:
        p_n_ratio = pos_score/neg_score
        n_p_ratio = neg_score/pos_score
    
    return [pos_score, neg_score, p_n_ratio, n_p_ratio]



#-----------------------------------------------------------------------------

# create files for emo and subj dictionaries
emo_dic = {}
subj_dic = {}
with open(os.path.join(this_directory,'Emotion-Lexicon-Dictionary.p'),"rb") as f_p:
	emo_dic = pickle.load(f_p, encoding='latin1')
	f_p.close()

with open(os.path.join(this_directory,'subjective_lexicon_dic.p'),"rb") as f_p:
	subj_dic = pickle.load(f_p, encoding='latin1')
	f_p.close()



def vote(tf_res, emo_res, ngram_res, pos_neg_res):

    result_votes = []
    for i in range(0,len(emo_res)):
        votes = {}
        votes[0]=0
        votes[1]=0
        votes[2]=0
        votes[3]=0
        votes[4]=0
        
        # whatever number tf_res voted for 
        votes[tf_res[i]] += 1
        votes[emo_res[i]] += 1
        votes[ngram_res[i]] += 1
        votes[pos_neg_res[i]] += 1
        
        most_votes=0
        most_popular = 0
        for key,val in votes.items():
            max_votes = max(val, most_votes)
            most_popular = key
            
        # everyone voted on something different
        if max_votes == 1:
            result_votes.append(tf_res[i])
        else: 
            result_votes.append(most_popular)
        

    return result_votes

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
    load_pos_neg()
    reader = csv.reader(csv_file)
    for idx,row in enumerate(reader):
        if idx == 0:
            continue
		
        cleaned_text = clean_text(row[2])
        phraseId = int(row[0])
        ids.append(phraseId)
        #print(phraseId)
        
        #emo_vector = score_emo(cleaned_text, emo_dic)
        #subj_vector = score_subj(cleaned_text, subj_dic)
        pos_neg_vector = score_pos_neg(cleaned_text.split())

        total_vector = []
        #total_vector.extend(emo_vector)
        #total_vector.extend(subj_vector)
        		
        #data_emo.append(total_vector)
        data_pos_neg.append(pos_neg_vector)
        #data_tf.append([cleaned_text])

process_data(csv_file)

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
#emo = clf_emo.predict(data_emo)
pos_neg = clf_pos_neg.predict(data_pos_neg)







