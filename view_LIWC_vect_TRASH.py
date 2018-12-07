#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:48:29 2018

@author: miaaltieri
"""
import os
import pickle

this_directory = os.getcwd()
with open(os.path.join(this_directory,"LIWC_vector"),"rb") as f_p:
    LIWC_vect = pickle.load(f_p)
    
for val in LIWC_vect:
    print(val)