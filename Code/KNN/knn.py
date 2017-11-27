import numpy as np
import os, sys, copy
from sklearn.preprocessing import normalize
import math
from pandas import DataFrame
import argparse

def loadData(filePath):
    # if file doesnt exist then return None
    if not os.path.exists(filePath):
        return None

    # load data from the file
    df = DataFrame.from_csv(filePath, sep='\s+', header=None, index_col=False)
    # print df

    obj_cols = df.select_dtypes(include=['object']).columns.values.tolist()

    for col in obj_cols:
        df[col] = df[col].astype('category')

    obj_cols = df.select_dtypes(['category']).columns

    df[obj_cols] = df[obj_cols].apply(lambda x: x.cat.codes)

    data = df.values

    return (data[:,:-1],data[:,-1])


def calc_knn(train_data,test_data,train_labels,test_labels,k):
    
    #Array to store the predicted results
    res = np.zeros(test_labels.shape[0])
    #Array to calculate the distance between the neighbors
    dist = np.zeros((test_data.shape[0],train_data.shape[0]))
    for i in range(test_data.shape[0]):
        for j in range(train_data.shape[0]):
            dist[i][j] = np.linalg.norm(test_data[i]-train_data[j])
    ind = np.zeros((dist.shape[0],k)).astype(int)
    #Capturing the first k indices
    for i in range(dist.shape[0]):
        ind[i] = np.argsort(dist[i])[:k]

    for i in range(test_labels.shape[0]):
        unique,counts = np.unique(train_labels[ind[i]],return_counts=True)
        grp = list(zip(counts,unique))
        #Handling cases where k is even and the neighbors have conflicting labels. k is reduced untill a clear winner is found out
        if(k % 2 == 0 and grp[0][0] == k/2):
            unique,counts = np.unique(train_labels[ind[i,:-1]],return_counts=True)
            grp = list(zip(counts,unique))
        
        grp.sort(reverse = True)
        res[i] = grp[0][1]
    
    a,b,c,d = 0,0,0,0

    #Finding the Accuracy, Precision, Recall and Fmeasure
    for i in range(test_labels.shape[0]):
        if(test_labels[i] == 1 and res[i] == 1):
            a += 1
        elif(test_labels[i] == 1 and res[i] == 0):
            b += 1
        elif(test_labels[i] == 0 and res[i] == 1):
            c += 1
        else:
            d += 1
    
    accuracy,precision,recall,fmeasure = (a+d)/(a+b+c+d),(a)/(a+c),(a)/(a+b),(2*a)/(2*a+b+c)
    
    return accuracy,precision,recall,fmeasure

def main():
	parser = argparse.ArgumentParser(description='KNN')
	requiredNamed = parser.add_argument_group('Required named arguments')
	requiredNamed.add_argument('-i', '--input', help='Input data file name', required=True, type=str)
	requiredNamed.add_argument('-n', '--num', help='Number of Neighbors', required=True, type=int)
	args = parser.parse_args()

	file_name = args.input
	k = args.num

	sum_accuracy,sum_precision,sum_recall,sum_fmeasure = np.zeros(4)
	#Loading data from input file
	data,labels = loadData(file_name)
	data = normalize(data, axis=0, norm='max')
	div_val = math.ceil(data.shape[0]/10)
	#Performing 10 cross validation and finding the average accuarcy, precision, recall and fmeasure for the dataset
	for i in range(1,11):
	    train_data,test_data = np.concatenate((data[:div_val*(i-1)],data[div_val*i:]), axis=0),data[div_val*(i-1):div_val*i]
	    train_labels,test_labels = np.concatenate((labels[:div_val*(i-1)],labels[div_val*i:]), axis=0),labels[div_val*(i-1):div_val*i]
	    accuracy,precision,recall,fmeasure = calc_knn(train_data,test_data,train_labels,test_labels,k)
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
