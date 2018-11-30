#
#

import csv
import sys, os
import numpy
import pickle


train_file = 'train.csv'
test_file = 'test.csv'

def isEmpty(x):
    if x:
        # print("true")
        return True;
    else:
        # print("false")
        return False;


# parse the file as a csv file, return an array of the outcome of the file
def parse_file(file):
    data = open(file, 'rt')
    reader = csv.reader(data, delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
    x = list(reader)
    # x is array of arrays
    x.sort(key=lambda x:x[1])
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
            print('isempty')
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
                    print('whilebreak')
                    break

        z.append(y)
        # print(y)
    # print(z)
    # return z
    return numpy.array(x)

def parse_pickle(top_commun_path):
    with open(top_commun_path,"rb") as f_p:
        top_100_commun = pickle.load(f_p)
        print(top_100_commun)


def main(*argv):
    # set args
    args=argv[0]
    print(args)
    # for i in range(1,len(args)):
        # train_data = parse_pickle(args[0])
        # print train_data
    Longest_Only = parse_file(train_file)
    this_directory = os.getcwd()
    with open(os.path.join(this_directory,"Longest_Only"),'wb') as out:
        pickle.dump(Longest_Only,out)
    # train_data = parse_pickle(args[1])
    # print(train_data)
    exit(0)

if __name__ == "__main__":
    main(sys.argv)
