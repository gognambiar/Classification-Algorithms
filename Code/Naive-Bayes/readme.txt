usage: naive_bayes.py [-h] [-i INPUT] [-t TRA] [-p TES]

Naive_Bayes

optional arguments:
  -h, --help            show this help message and exit

Required named arguments:
  -i INPUT, --input INPUT
                        Input data file name
  -t TRA, --tra TRA     Training data set file name
  -p TES, --tes TES     Testing data set file name

This program can run in 2 ways - 

Train and test data sets are given seperately - 
sample command - python naive_bayes.py -i project3_dataset1.txt
it returns accuracy, precision, recall and f-measure

Single data set is given - 
sample command - python naive_bayes.py -t project3_dataset4.txt -p project3_dataset4_test.txt
prints the class posterior probability of all the labels

Formats
	
	-t : 	string : 	Training data set file name

	-p : 	string :	Testing data set file name 
  
  -i :  string : Input data file name for single file format for 10-fold cross validation

Required Libraries:
import pandas as pd
import operator
import os, sys, copy
import numpy as np
import math
import argparse
