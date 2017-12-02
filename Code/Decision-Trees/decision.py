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
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def loadData(filePath):
    # function to load data from file

    # if file doesnt exist then return None
    if not os.path.exists(filePath):
        return None

    mappings = {}

    # load data from the file
    df = DataFrame.from_csv(filePath, sep='\s+', header=None, index_col=None)

    obj_cols = df.select_dtypes(include=['object']).columns.values.tolist()
    # print obj_cols
    # print df.dtypes
    # print df[0]
    # exit(0)

    for col in obj_cols:
        # print df[col].cat_column.dtype == 'category'
        # df[col] = df[col].astype('category')
        # df[col] = pd.Categorical(df[col])
        # print pd.factorize(df[col])[1]
        coded, index = pd.factorize(df[col])
        index = index.tolist()
        df[col] = coded
        mappings[col] = index
    # exit(0)

    obj_cols = df.select_dtypes(['category']).columns

    df[obj_cols] = df[obj_cols].apply(lambda x: x.cat.codes)

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
        # print self.data.shape
        self.root = self.createTree(self.data)

        print simplejson.dumps(self.root)
        # exit(0)

    def getGini(self, data, classes):
        gini = 1
        for val in classes:
            gini -= (np.count_nonzero(data[:,-1] == val) / float(data.shape[0])) ** 2

        return gini


    def getSplit(self, data, gini):
        selIndex = None
        selValue = None
        gain = None
        selLeft = None
        selRight = None
        selGiniLeft = None
        selGiniRight = None


        randomFeatureIndices = np.random.choice(xrange(data.shape[1]-1), int(np.ceil((data.shape[1]-1)*self.numFeatures)), replace=False)
        # print randomFeatureIndices
        # exit(0)
        for index in randomFeatureIndices:
            values = set(data[:,index].reshape(1,-1).tolist()[0])
            for value in values:
                # print index,value,values
                left = data[data[:, index] < value]
                right = data[data[:, index] >= value]
                
                if left.shape[0] == 0:
                    giniLeft = 0
                else:
                    giniLeft = self.getGini(left, set(left[:,-1].reshape(1,-1).tolist()[0]))
                if right.shape[0] == 0:
                    giniRight = 0
                else:
                    giniRight = self.getGini(right, set(right[:,-1].reshape(1,-1).tolist()[0]))

                # print left.shape,giniLeft
                # print right.shape, giniRight

                infoGain = gini - (giniLeft*(left.shape[0]/float(data.shape[0]))) - (giniRight*(right.shape[0]/float(data.shape[0])))

                # print infoGain,gini,giniLeft,left.shape[0],giniRight,right.shape[0]

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
        # print data.shape,depth,self.minRows,self.maxDepth
        gini = self.getGini(data,set(data[:,-1].reshape(1,-1).tolist()[0]))
        root = {'data':data.tolist(), 'gini':gini, 'dataCount': data.shape[0]}
        if gini == 0 or data.shape[0] < self.minRows or depth >= self.maxDepth:
            root['type'] = 'terminal'
            # print data.shape
            unique, counts = np.unique(data[:,-1], return_counts=True)
            # print unique,counts
            count = zip(unique, counts)
            count = sorted(count, key = lambda x: x[1], reverse=True)
            # print count
            root['label'] = count[0][0]
            # exit(0)
            # root['label'] = 
            # print '\t'*depth,data
            # return root
        else:
            root['type'] = 'split'
            index, value, gain, left, right = self.getSplit(data, gini)
            if left is not None and right is not None and left.shape[0] != 0 and right.shape[0] != 0:
                root['index'] = index
                root['value'] = value
                if index in self.mappings:
                    root['actualValue'] = self.mappings[index][value]
                root['left'] = self.createTree(left,depth+1)
                root['right'] = self.createTree(right,depth+1)
                root['gain'] = gain
                del root['data']
            else:
                root['type'] = 'terminal'
                # print data.shape
                unique, counts = np.unique(data[:,-1], return_counts=True)
                # print unique,counts
                count = zip(unique, counts)
                count = sorted(count, key = lambda x: x[1], reverse=True)
                # print count
                root['label'] = count[0][0]
        return root
        # if gini = 

    def predictUtil(self, row):
        temp = self.root
        while True:
            if temp['type'] == 'terminal':
                return temp['label']
            index, val = temp['index'],temp['value']
            if row[index] < val:
                temp = temp['left']
            else:
                temp = temp['right']


    def predict(self,data):
        labels = np.apply_along_axis(self.predictUtil, 1, data)
        return labels
        

def KFoldCrossValidation(classifier, data, labels, k=1):
    data = np.hstack((data,labels.reshape(-1,1)))

    # np.random.shuffle(data)

    chunks = np.array_split(data, k)

    accuracy = np.zeros((len(chunks),1))
    precision = np.zeros((len(chunks),1))
    recall = np.zeros((len(chunks),1))
    f1_score = np.zeros((len(chunks),1))

    for i in xrange(k):
        temp = copy.copy(chunks)
        test = temp[i]
        del temp[i]
        train = np.vstack(temp)
        # print train.shape,test.shape
        classifier.fit(train[:,:-1],train[:,-1])
        pred = classifier.predict(test[:,:-1])
        # print pred.shape,test[:,:-1].shape
        # predicted[i] = np.sum(pred == test[:,-1])/ float(pred.shape[0])

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for j in xrange(pred.shape[0]):
            if pred[j] == test[j][-1] and pred[j] == 1:
                true_positive += 1
            elif pred[j] == test[j][-1] and pred[j] == 0:
                true_negative += 1
            elif pred[j] == 1:
                false_positive += 1
            else:
                false_negative += 1

        accuracy[i] = (true_negative + true_positive) / float(pred.shape[0])
        precision[i] = true_positive / float(true_positive + false_positive)
        recall[i] = true_positive / float(true_positive + false_negative)
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
    parser.add_argument('-d', '--maxDepth', help='Maximum Depth of Decision Tree', type=int, default=100000)
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

    # data = np.array([[2,3,1],[4,5,0],[6,7,1],[8,9,0],[10,11,1]])
    # print data.shape
    # print labels.shape

    tree = DecisionTree(maxDepth,minRows,numFeatures=1)
    tree.fit(data,labels, mappings)
    # predicted =  tree.predict(data)
    # print np.sum(predicted == labels)/ float(labels.shape[0])
    # exit(0)
    # print KFoldCrossValidation(tree,data,labels,k=10)

    # clf = DecisionTreeClassifier(max_depth=maxDepth)
    # clf.fit(data, labels)
    # tree.export_graphviz(clf, out_file='tree.dot')      
    # print np.mean(cross_val_score(clf, data, labels, cv=10))



if __name__ == "__main__":
    main(sys.argv[1:])
