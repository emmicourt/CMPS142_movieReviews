#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:52:35 2018

@author: miaaltieri
"""
import os
import pickle

result_dic = {}
mapping = {}
    

# open file
this_directory = os.getcwd()
LIWCLoc = this_directory +'/CleanedData/Data_by_Rating'

# go through line by line
input_file = open('LIWC2007.txt')
try:
    for i, line in enumerate(input_file):
        tokens = line.split()
        # print(line)
        if(len(tokens) > 0):
            
            # when the line starts with a number then we add that entry to our
            # dictionary, this is the first 100 lines-ish
            if tokens[0].isdigit():
                result_dic[tokens[1]] = []
                mapping[tokens[0]] = tokens[1]
                print("adding ",tokens[1],"to dic")
                
            # when the line starts with a word/regex it means that we are adding
            # words into our dictionary values
            if tokens[0].isalpha() or tokens[0][len(tokens[0])-1] == '*' :                
                
                # gather all the numbers and place in the correct dictionary
                # based on the mapping
                for j in range (1,len(tokens)):
                    if tokens[0] == 'kind':
                        print(tokens[j])
                        
                    if tokens[0] == 'like':
                        print(tokens[j])
                        
                    if tokens[j] not in mapping:
                        #print("ERROR: ",tokens[j],"not in dict for word:",tokens[0])
                        continue
   
                    key = mapping[tokens[j]]
                    result_dic[key].append(tokens[0])
                     
#                    if tokens[0][len(tokens[0])-1] == '*' :
#                        print(key,tokens[0])
                    
        
finally:
    input_file.close()
    
    
# handling errors with word 'kind' 
key = mapping['131']
if 'kind' not in result_dic[key]:
    result_dic[key].append('kind')

key = mapping['125']
if 'kind' not in result_dic[key]:
    result_dic[key].append('kind')

key = mapping['135']
if 'kind' not in result_dic[key]:
    result_dic[key].append('kind')

key = mapping['126']
if 'kind' not in result_dic[key]:
    result_dic[key].append('kind')


# handling errors with word 'like'
key = mapping['125']
if 'like' not in result_dic[key]:
    result_dic[key].append('like')

key = mapping['464']
if 'like' not in result_dic[key]:
    result_dic[key].append('like')

key = mapping['126']
if 'like' not in result_dic[key]:
    result_dic[key].append('like')

key = mapping['253']
if 'like' not in result_dic[key]:
    result_dic[key].append('like')

this_directory = os.getcwd()
with open(os.path.join(this_directory,"LIWC_dict"),'wb') as out:
    pickle.dump(result_dic,out)

"""   
for key,val in result_dic.items():
    print(key,val)
"""

