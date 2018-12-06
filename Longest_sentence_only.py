#
#
import csv
import sys, os
import numpy
import pickle
import nltk
from nltk import word_tokenize
import string
from nltk.corpus import stopwords 

train_file = 'train.csv'
test_file = 'test.csv'


#Initialize text cleaning modules
lemma = nltk.wordnet.WordNetLemmatizer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def isEmpty(x):
    if x:
        # print("true")
        return True
    else:
        # print("false")
        return False

# this cleans the text by:
#   putting everything to lowercase
#   removing punctation
#   lemmatizing
#   removing stopwords 
def clean_text (text):
    text = text.translate(remove_punctuation_map).lower()
    word_tokens = word_tokenize(text) 
    lemmatized_sentence = [lemma.lemmatize(word.lower()) 
        for word in word_tokens if word.isalpha()]
    space = ' '
    sentence = space.join(lemmatized_sentence)
    return sentence


# parse the file as a csv file, return an array of the outcome of the file
def parse_file(file):
    data = open(file, 'rt')
    reader = csv.reader(data, delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
    x = list(reader)
    # x is array of arrays
    x.sort(key=lambda x:x[1])
    # print(x)
    i=0
    z=[]
    y=[]
    while isEmpty(x):
        # print('inloop')
        # print(x[i]);

        try:
            y=x[i]
            x.pop(i)
        except IndexError:
            print('break')
            break
        # print(y)
        # i+=1;
        if not isEmpty(x):
            # print('isempty')
            z.append(y)
            break
        else:
            # print('notempty')
            while isEmpty(x) &(x[i][1] == y[1]):
                if len(x[i][2]) > len(y[2]):
                    y=x.pop(i)
                    # print(y)
                else:
                    x.pop(i)
                # i+=1;
                if not isEmpty(x):
                    # print('whilebreak')
                    break

        z.append(y)

    for i,row in enumerate(z):
        z[i][2] = clean_text(row[2])
        
    return numpy.array(z)

def parse_pickle(top_commun_path):
    with open(top_commun_path,"rb") as f_p:
        top_100_commun = pickle.load(f_p)
        print(top_100_commun)


def main(*argv):
    # set args
    args=argv[0]
    # for i in range(1,len(args)):
        # train_data = parse_pickle(args[0])
        # print train_data
    Longest_Only = parse_file(train_file)
    this_directory = os.getcwd()
    # print(Longest_Only)
    with open(os.path.join(this_directory,"Longest_Only"),'wb') as out:
        pickle.dump(Longest_Only,out)
    # train_data = parse_pickle(args[1])
    # print(train_data)
    exit(0)

if __name__ == "__main__":
    main(sys.argv)
