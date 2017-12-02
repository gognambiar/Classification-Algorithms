import numpy as np
from pandas import DataFrame
import pandas as pd
# from sklearn.metrics.pairwise import euclidean_distances
import os
import sys
import getopt
import copy
import argparse
import simplejson
import copy

def loadData(filePath):
    # function to load data from file

    # if file doesnt exist then return None
    if not os.path.exists(filePath):
        return None

    mappings = {}

    # load data from the file
    df = DataFrame.from_csv(filePath, sep='\s+', header=None, index_col=None)

    # load data from the file
    obj_cols = df.select_dtypes(include=['object']).columns.values.tolist()

    # convert cateogrical features to numeric codes
    for col in obj_cols:
        coded, index = pd.factorize(df[col])
        index = index.tolist()
        df[col] = coded
        mappings[col] = index
   

    data = df.values


    # extract labels and data
    labels = data[:,-1]
    data = data[:,:-1]

    print mappings,'\n\n'

    # exit(0)

    return (data,labels, mappings)


class DecisionTree(object):

    def __init__(self, maxDepth=1, minRows=1, numFeatures=1.0):
        self.maxDepth = maxDepth
        self.minRows = minRows
        self.numFeatures = numFeatures
        self.root = None

    def fit(self, data, labels, mappings = {}):
        self.data = data
        self.labels = labels
        self.mappings = mappings
        self.data = np.hstack((data,labels.reshape(-1,1)))
        self.root = self.createTree(self.data)

        # print simplejson.dumps(self.root)

    def getGini(self, data, classes):
        # function to calculate gini for the node
        gini = 1
        for val in classes:
            gini -= (np.count_nonzero(data[:,-1] == val) / float(data.shape[0])) ** 2

        return gini


    def getSplit(self, data, gini):
        # function to get best split using gini and information gain
        selIndex = None
        selValue = None
        gain = None
        selLeft = None
        selRight = None
        selGiniLeft = None
        selGiniRight = None

        # get random features for splitting 
        randomFeatureIndices = np.random.choice(xrange(data.shape[1]-1), int(np.ceil((data.shape[1]-1)*self.numFeatures)), replace=False)
        
        for index in randomFeatureIndices:
            # get unique values in the feature
            values = set(data[:,index].reshape(1,-1).tolist()[0])

            # for each value get left, right and information gain
            for value in values:
                left = data[data[:, index] < value]
                right = data[data[:, index] >= value]
                
                # if left is empty
                if left.shape[0] == 0:
                    giniLeft = 0
                else:
                    giniLeft = self.getGini(left, set(left[:,-1].reshape(1,-1).tolist()[0]))

                # if right is empty
                if right.shape[0] == 0:
                    giniRight = 0
                else:
                    giniRight = self.getGini(right, set(right[:,-1].reshape(1,-1).tolist()[0]))

                # calculate information gain
                infoGain = gini - (giniLeft*(left.shape[0]/float(data.shape[0]))) - (giniRight*(right.shape[0]/float(data.shape[0])))


                # if info gain is max then store
                if gain is None or infoGain > gain:
                    selIndex = index
                    selValue = value
                    gain = infoGain
                    selLeft = left
                    selRight = right
                    selGiniLeft = giniLeft
                    selGiniRight = giniRight

        return (selIndex, selValue, gain, selLeft, selRight)


    def createTree(self, data, depth=0):
        # function to create tree

        # calculate gini for parent
        gini = self.getGini(data,set(data[:,-1].reshape(1,-1).tolist()[0]))
        root = {'data':data.tolist(), 'gini':gini, 'dataCount': data.shape[0]}

        # if no more splitting possible then create terminal
        if gini == 0 or data.shape[0] < self.minRows or depth >= self.maxDepth:
            root['type'] = 'terminal'

            # calculate label for node by majority
            unique, counts = np.unique(data[:,-1], return_counts=True)
            count = zip(unique, counts)
            count = sorted(count, key = lambda x: x[1], reverse=True)
            root['label'] = count[0][0]
        
        # else split further    
        else:
            root['type'] = 'split'

            # get best split
            index, value, gain, left, right = self.getSplit(data, gini)

            # if left or right is empty then make it terminal
            if left is not None and right is not None and left.shape[0] != 0 and right.shape[0] != 0:
                root['index'] = index
                root['value'] = value
                if index in self.mappings:
                    root['actualValue'] = self.mappings[index][value]

                # create left subtree
                root['left'] = self.createTree(left,depth+1)

                # create right subtree
                root['right'] = self.createTree(right,depth+1)
                root['gain'] = gain
                del root['data']
            else:
                root['type'] = 'terminal'
                unique, counts = np.unique(data[:,-1], return_counts=True)
                count = zip(unique, counts)
                count = sorted(count, key = lambda x: x[1], reverse=True)
                root['label'] = count[0][0]
        return root

    def predictUtil(self, row):
        # helper function for prediction 
        temp = self.root
        while True:
            # if terminal then get label 
            if temp['type'] == 'terminal':
                return temp['label']
            index, val = temp['index'],temp['value']
            # else if less then recur on left
            if row[index] < val:
                temp = temp['left']
            # else right
            else:
                temp = temp['right']


    def predict(self,data):
        labels = np.apply_along_axis(self.predictUtil, 1, data)
        return labels
        

def KFoldCrossValidation(classifier, data, labels, k=1):
    # function to calculate accuracy, precision, recall and f1-score for K-fold cross validation
    data = np.hstack((data,labels.reshape(-1,1)))

    # split data to folds
    chunks = np.array_split(data, k)

    # arrays to store a,p,r,f1
    predicted = np.zeros((len(chunks),1))
    accuracy = np.zeros((len(chunks),1))
    precision = np.zeros((len(chunks),1))
    recall = np.zeros((len(chunks),1))
    f1_score = np.zeros((len(chunks),1))

    # k-fold
    for i in xrange(k):
        # get test and train 
        temp = copy.copy(chunks)
        test = temp[i]
        del temp[i]
        train = np.vstack(temp)
        # print train.shape,test.shape
        classifier.fit(train[:,:-1],train[:,-1])
        pred = classifier.predict(test[:,:-1])
        # print pred.shape,test[:,-1].shape
        predicted[i] = np.sum(pred == test[:,-1].reshape(-1,1))/ float(pred.shape[0])
        # print predicted[i]

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for j in xrange(pred.shape[0]):
            if pred[j][0] == test[j][-1] and pred[j][0] == 1:
                true_positive += 1
            elif pred[j][0] == test[j][-1] and pred[j][0] == 0:
                true_negative += 1
            elif pred[j][0] == 1:
                false_positive += 1
            else:
                false_negative += 1

        accuracy[i] = (true_negative + true_positive) / float(pred.shape[0])
        try:
            precision[i] = true_positive / float(true_positive + false_positive)
        except:
            precision[i] = 0.0
        try:
            recall[i] = true_positive / float(true_positive + false_negative)
        except:
            recall[i] = 0.0
        if recall[i] != 0.0 and precision[i] != 0.0:
            f1_score[i] = (2*precision[i]*recall[i]) / (precision[i] + recall[i])
        print '*'*80
        print 'Accuracy:\t',accuracy[i]
        print 'Precision:\t',precision[i]
        print 'Recall:\t',recall[i]
        print 'F1-Score:\t',f1_score[i]
        print '*'*80

    return np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(f1_score)


def main(argv):
    # arguments
    parser = argparse.ArgumentParser(description='Decision Tree Classifier')

    # optional arguments
    parser.add_argument('-d', '--maxDepth', help='Maximum Depth of Decision Tree', type=int, default=10000)
    parser.add_argument('-r', '--minRows', help='Minimum Rows required to split', type=int, default=1)
    # parser.add_argument('-o', '--output', help='Output file to store PCA visualization')

    # required arguments
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input data file name', required=True, type=str)
    
    args = parser.parse_args()
    
    # parse arguments
    inputFile = args.input
    maxDepth = args.maxDepth
    minRows = args.minRows

    # print maxDepth,minRows


    # load initial data
    data,labels, mappings = loadData(inputFile)


    tree = DecisionTree(maxDepth,minRows,numFeatures=1)
    # tree.fit(data,labels, mappings)
    # predicted =  tree.predict(data)
    # print np.sum(predicted == labels)/ float(labels.shape[0])
    # exit(0)
    print KFoldCrossValidation(tree,data,labels,k=10)





if __name__ == "__main__":
    main(sys.argv[1:])
