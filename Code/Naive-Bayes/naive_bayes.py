import pandas as pd
import operator
import os, sys, copy
import numpy as np
from sklearn.preprocessing import normalize
import math
import argparse

#Function to calculate gaussian probability
def gaussian_prob(mean_val,var_val,x_val):
    return (1/((2*np.pi*var_val)**(1/2)))*(np.exp(-((x_val-mean_val)**2)/(2*var_val)))

#Function to calculate categorical probability
def categ_prob(x_val,i,dct):
    return dct[i][x_val]


def loadData(filePath):
    # if file doesnt exist then return None
    if not os.path.exists(filePath):
        return None

    # load data from the file
    df = pd.DataFrame.from_csv(filePath, sep='\s+', header=None, index_col=None)
    # print df

    obj_cols = df.select_dtypes(include=['object']).columns.values.tolist()

    for col in obj_cols:
        df[col] = df[col].astype('category')

    obj_cols = df.select_dtypes(['category']).columns

    df[obj_cols] = df[obj_cols].apply(lambda x: x.cat.codes)

    data = df.values

    return (data,obj_cols)

def calc_naive_bayes(train_data,test_data,test_labels,obj_cols):
	#Dividing the training set into values with 1 and 0
    data_yes = np.asarray([train_data[i] for i in range(len(train_data)) if train_data[i][-1] == 1])[:,:-1]
    data_no = np.asarray([train_data[i] for i in range(len(train_data)) if train_data[i][-1] == 0])[:,:-1]
    data_yes = pd.DataFrame(data_yes)
    data_no = pd.DataFrame(data_no)
    gaus_yes,gaus_no = [],[]


    #Handling categorical columns
    dct_yes,dct_no = {},{}
    
    if(len(obj_cols) > 0):
        for k in obj_cols:
            cat_prob_yes,cat_prob_no = {},{}
            for j in data_yes[k]:
                cat_prob_yes[j] = cat_prob_yes.get(j,0) + 1
            for j in data_no[k]:
                cat_prob_no[j] = cat_prob_no.get(j,0) + 1
                
            for i in cat_prob_yes:
                cat_prob_yes[i] = cat_prob_yes[i]/data_yes.shape[0]
    
            for i in cat_prob_no:
                cat_prob_no[i] = cat_prob_no[i]/data_no.shape[0]
            dct_yes[k] = cat_prob_yes
            dct_no[k] = cat_prob_no
    
    #Calculating the class probabilities
    class_prob_yes,class_prob_no = 0,0
    for i in range(test_labels.shape[0]):
        if(test_labels[i] == 1):
            class_prob_yes += 1
        else:
            class_prob_no += 1
    

    for i in range(data_yes.shape[1]):
        gaus_yes.append((data_yes[i].mean(axis = 0),data_yes[i].var(axis = 0)))
    for i in range(data_no.shape[1]):
        gaus_no.append((data_no[i].mean(axis = 0),data_no[i].var(axis = 0)))
    res = np.zeros(test_data.shape[0])
    for i in range(test_data.shape[0]):
        prob_yes,prob_no = 1,1
        #Calculating the Descriptor Posterior Probability
        for j in range(len(test_data[i])):
            if(j in obj_cols):
                prob_yes *= categ_prob(test_data[i][j],j,dct_yes)
                prob_no *= categ_prob(test_data[i][j],j,dct_no)
            else:
                prob_yes *= gaussian_prob(gaus_yes[j][0],gaus_yes[j][1],test_data[i][j])
                prob_no *= gaussian_prob(gaus_no[j][0],gaus_no[j][1],test_data[i][j])
        #Comparing the posteriori probability for both the classes
        if(prob_yes*(data_yes.shape[0]/train_data.shape[0]) > prob_no*(data_no.shape[0]/train_data.shape[0])):
            res[i] = 1
        else:
            res[i] = 0

    a,b,c,d = 0,0,0,0
            
    for i in range(test_labels.shape[0]):
        if(test_labels[i] == 1 and res[i] == 1):
            a += 1
        elif(test_labels[i] == 1 and res[i] == 0):
            b += 1
        elif(test_labels[i] == 0 and res[i] == 1):
            c += 1
        else:
            d += 1
    

    if(a == 0):
        accuracy,precision,recall,fmeasure = (d)/(b+c+d),0,0,0
    else:    
        accuracy,precision,recall,fmeasure = (a+d)/(a+b+c+d),(a)/(a+c),(a)/(a+b),(2*a)/(2*a+b+c)

    
    return accuracy,precision,recall,fmeasure

def main():
	parser = argparse.ArgumentParser(description='Naive_Bayes')
	requiredNamed = parser.add_argument_group('Required named arguments')
	requiredNamed.add_argument('-i', '--input', help='Input data file name', required=True, type=str)
	args = parser.parse_args()

	file_name = args.input
	sum_accuracy,sum_precision,sum_recall,sum_fmeasure = np.zeros(4)
	#Loading data from input file
	data,obj_cols = loadData(file_name)
	data = normalize(data, axis=0, norm='max')
	div_val = math.ceil(data.shape[0]/10)
	#Performing 10 cross validation and finding the average accuarcy, precision, recall and fmeasure for the dataset
	for i in range(1,11):
	    train_data,test_data = np.concatenate((data[:div_val*(i-1)],data[div_val*i:]), axis=0),data[div_val*(i-1):div_val*i,:-1]
	    test_labels = data[div_val*(i-1):div_val*i,-1]
	    accuracy,precision,recall,fmeasure = calc_naive_bayes(train_data,test_data,test_labels,obj_cols)
	    
	    sum_accuracy += accuracy
	    sum_precision += precision
	    sum_recall += recall
	    sum_fmeasure += fmeasure

	print("The Accuracy is "+ str(sum_accuracy*10))
	print("The Precision is "+ str(sum_precision*10))
	print("The Recall is "+ str(sum_recall*10))
	print("The F-Measure is "+ str(sum_fmeasure*10))

if __name__ == '__main__':
	main()
