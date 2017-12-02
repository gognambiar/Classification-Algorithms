import numpy as np
import os, sys, copy
import math
from pandas import DataFrame
from pandas import factorize
import argparse
from sklearn import preprocessing

#Calculate euclidean distance, handles both categorical and continuous data
def euclidean_dist(a,b,obj_cols):
	dist = 0
	for i in range(a.shape[0]):
		if(i in obj_cols):
			if(a[i] != b[i]):
				dist += 1
		else:
			dist += (a[i]-b[i])**2
	
	return dist**(1/2)

#Load data from file name provided
def loadData(filePath):
	# function to load data from file

	# if file doesnt exist then return None
	if not os.path.exists(filePath):
		return None

	mappings = {}

	# load data from the file
	df = DataFrame.from_csv(filePath, sep='\s+', header=None, index_col=None)
	#Record columns which have categorical data
	obj_cols = df.select_dtypes(include=['object']).columns.values.tolist()

	for col in obj_cols:
		coded, index = factorize(df[col])
		index = index.tolist()
		df[col] = coded
		mappings[col] = index

	data = df.values

	return (data[:,:-1],data[:,-1],obj_cols)


def calc_knn(train_data,test_data,train_labels,test_labels,k,obj_cols):
	
	#Array to store the predicted results
	res = np.zeros(test_labels.shape[0])
	#Array to calculate the distance between the neighbors
	dist = np.zeros((test_data.shape[0],train_data.shape[0]))
	for i in range(test_data.shape[0]):
		for j in range(train_data.shape[0]):
			dist[i][j] = euclidean_dist(test_data[i],train_data[j],obj_cols)
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
	
	if(a == 0):
		accuracy,precision,recall,fmeasure = (d)/(b+c+d),0,0,0
	else:
		accuracy,precision,recall,fmeasure = (a+d)/(a+b+c+d),(a)/(a+c),(a)/(a+b),(2*a)/(2*a+b+c)
	
	return accuracy,precision,recall,fmeasure

def main():
	parser = argparse.ArgumentParser(description='KNN')
	requiredNamed = parser.add_argument_group('Required named arguments')
	#optional argument
	requiredNamed.add_argument('-n', '--num', help='Number of Neighbors', type=int)
	requiredNamed.add_argument('-t', '--tra', help='Training data set file name', type=str)
	requiredNamed.add_argument('-p', '--tes', help='Testing data set file name', type=str)
	requiredNamed.add_argument('-i', '--input', help='Input data file name', type=str)


	args = parser.parse_args()
	#Handle the condition where both train and test data set are given seperately
	if(args.tra and args.tes):
		if(args.num):
			train_data, train_labels,obj_cols = loadData(args.tra)
			test_data, test_labels,obj_cols = loadData(args.tes)
			k = args.num
			scaler = preprocessing.StandardScaler().fit(train_data)
			norm_train_data = scaler.transform(train_data)
			norm_test_data = scaler.transform(test_data)
			accuracy,precision,recall,fmeasure = calc_knn(norm_train_data,norm_test_data,train_labels,test_labels,k,obj_cols)
			print("The Accuracy is "+ str(accuracy*100))
			print("The Precision is "+ str(precision*100))
			print("The Recall is "+ str(recall*100))
			print("The F-Measure is "+ str(fmeasure*100))			
		else:
			print("Please enter K")
			sys.exit(2)


	
	#If k is provided by the user and a single dataset is given
	elif(args.num):
		file_name = args.input
		k = args.num
		sum_accuracy,sum_precision,sum_recall,sum_fmeasure = np.zeros(4)
		#Loading data from input file
		data,labels,obj_cols = loadData(file_name)
		#data = normalize(data, axis=0, norm='max')
		div_val = math.ceil(data.shape[0]/10)
		#Performing 10 cross validation and finding the average accuarcy, precision, recall and fmeasure for the dataset
		for i in range(1,11):
			train_data,test_data = np.concatenate((data[:div_val*(i-1)],data[div_val*i:]), axis=0),data[div_val*(i-1):div_val*i]
			train_labels,test_labels = np.concatenate((labels[:div_val*(i-1)],labels[div_val*i:]), axis=0),labels[div_val*(i-1):div_val*i]
			scaler = preprocessing.StandardScaler().fit(train_data)
			norm_train_data = scaler.transform(train_data)
			norm_test_data = scaler.transform(test_data)
			accuracy,precision,recall,fmeasure = calc_knn(norm_train_data,norm_test_data,train_labels,test_labels,k,obj_cols)
			sum_accuracy += accuracy
			sum_precision += precision
			sum_recall += recall
			sum_fmeasure += fmeasure

		print("The Accuracy is "+ str(sum_accuracy*10))
		print("The Precision is "+ str(sum_precision*10))
		print("The Recall is "+ str(sum_recall*10))
		print("The F-Measure is "+ str(sum_fmeasure*10))

	#If k is not provided by the user, the algorithm is run for k values uptill 10 and the k with best accuracy is returned
	else:
		file_name = args.input
		print("k value not provided in input and hence running for optimum k")
		opt_accuracy = 0
		for k in range(1,10):
			sum_accuracy,sum_precision,sum_recall,sum_fmeasure = np.zeros(4)
			#Loading data from input file
			data,labels,obj_cols = loadData(file_name)
			#data = normalize(data, axis=0, norm='max')
			div_val = math.ceil(data.shape[0]/10)
			#Performing 10 cross validation and finding the average accuarcy, precision, recall and fmeasure for the dataset
			for i in range(1,11):
				train_data,test_data = np.concatenate((data[:div_val*(i-1)],data[div_val*i:]), axis=0),data[div_val*(i-1):div_val*i]
				train_labels,test_labels = np.concatenate((labels[:div_val*(i-1)],labels[div_val*i:]), axis=0),labels[div_val*(i-1):div_val*i]
				scaler = preprocessing.StandardScaler().fit(train_data)
				norm_train_data = scaler.transform(train_data)
				norm_test_data = scaler.transform(test_data)
				accuracy,precision,recall,fmeasure = calc_knn(norm_train_data,norm_test_data,train_labels,test_labels,k,obj_cols)
				sum_accuracy += accuracy
				sum_precision += precision
				sum_recall += recall
				sum_fmeasure += fmeasure

			if(sum_accuracy > opt_accuracy):
				opt_accuracy = sum_accuracy
				opt_precision = sum_precision
				opt_recall = sum_recall
				opt_fmeasure = sum_fmeasure
				k_val = k

		print("The optimum values are for k = "+str(k_val))
		print("The Accuracy is "+ str(opt_accuracy*10))
		print("The Precision is "+ str(opt_precision*10))
		print("The Recall is "+ str(opt_recall*10))
		print("The F-Measure is "+ str(opt_fmeasure*10))

if __name__ == '__main__':
	main()
