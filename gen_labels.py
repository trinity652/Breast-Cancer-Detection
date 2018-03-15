#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 18 12:50:35 2017

@author: abhilasha
"""




import csv
import numpy as np 


text_file = csv.reader(open('labels_init.txt'), delimiter=" ")

f = open('labels.txt','w')

for row in text_file:
	if row[2] == 'NORM':
		f.write('0\n')
	else:
		f.write('1\n')

f.close()

