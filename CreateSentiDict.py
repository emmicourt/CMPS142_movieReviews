#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 23:04:11 2018

@author: miaaltieri
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:52:35 2018

@author: miaaltieri
"""
import os
import pickle

sentiment = {}

# open file
this_directory = os.getcwd()

# go through line by line
input_file = open('SentiWords_1.1.txt')
try:
    print(input_file)
    for i, line in enumerate(input_file):
        tokens = line.split()
        word = tokens[0][0:-2]
        val = tokens[1]
        sentiment[word]=val
        
finally:
    input_file.close()

this_directory = os.getcwd()
with open(os.path.join(this_directory,"Senti_dict"),'wb') as out:
    pickle.dump(sentiment,out)

"""   
for key,val in result_dic.items():
    print(key,val)
"""

