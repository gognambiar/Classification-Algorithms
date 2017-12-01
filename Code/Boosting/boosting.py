from decision import loadData,DecisionTree,KFoldCrossValidation
import getopt
import argparse
import os
import sys
import numpy as np
from copy import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class BoostedClassifier(object):
    def __init__(self, classifier, numIters=50, ratio=0.632):
        self.classifier = classifier
        self.numIters = numIters
        self.ratio = ratio

    def createSubsets(self, data, weights):
        length = data.shape[0]
        ratio = self.ratio

        originalDataLength = int(length*ratio)

        randomIndices = np.random.choice(length, originalDataLength, weights, replace=False)
        dataSets[i] = data[randomIndices]
        randomDuplicating = np.random.choice(dataSets[i].shape[0], length - originalDataLength, replace=True)
        dups = dataSets[i][randomDuplicating]
        dataSet = np.vstack((dataSets[i], dups))

        return dataSet

    def fit(self, data, labels):
        self.data = np.hstack((data,labels.reshape(-1,1)))
        numIters = self.numIters
        classifier = self.classifier

        weights = np.array([1/float(data.shape[0])]*data.shape[0]).reshape(-1,1)
        tempClassifier = copy(classifier)

        for i in xrange(numIters):




    def trainClassifiers(self):
        numClassifiers = self.numClassifiers
        dataSets = self.dataSets
        classifier = self.classifier

        classifiers = [None]*numClassifiers

        for i in xrange(numClassifiers):
            clf = copy(classifier)
            clf = train(clf, dataSets[i])
            classifiers[i] = clf

        self.classifiers = classifiers

    def train(self, classifier, data):

        


        for i in xrange(numIters):


    def getMostCount(self,data):
        # print 'data = ',data
        # print data
        unique, counts = np.unique(data, return_counts=True)
        # print unique,counts
        count = zip(unique, counts)
        count = sorted(count, key = lambda x: x[1], reverse=True)
        # print count
        return count[0][0]

    def predict(self, data):
        # print self.classifiers
        labelArr = [i.predict(data) for i in self.classifiers]
        # print labelArr
        labelArr = np.concatenate(list(zip(*labelArr))).reshape(-1,self.numClassifiers)
        # print labelArr.shape
        labels = np.apply_along_axis(self.getMostCount, 1, labelArr)
        return labels




def main(argv):
    # arguments
    parser = argparse.ArgumentParser(description='Random Forest Classifier')

    # optional arguments
    parser.add_argument('-d', '--maxDepth', help='Maximum Depth of Decision Tree', type=int, default=10)
    parser.add_argument('-r', '--minRows', help='Minimum Rows required to split', type=int, default=1)
    # parser.add_argument('-n', '--numClassifiers', help='Number of Classifiers', type=int, default=100)
    parser.add_argument('-x', '--ratio', help='Ratio of data for bagging', type=float, default=1-1/np.e)
    parser.add_argument('-r', '--numIters', help='Number of iterations', type=float, default=50)
    parser.add_argument('-f', '--numFeatures', help='Ratio of number of features for decision tree', type=float, default=0.2)
    # parser.add_argument('-o', '--output', help='Output file to store PCA visualization')

    # required arguments
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input data file name', required=True, type=str)
    
    args = parser.parse_args()
    
    # parse arguments
    inputFile = args.input
    maxDepth = args.maxDepth
    minRows = args.minRows
    # numClassifiers = args.numClassifiers
    ratio = args.ratio
    numFeatures = args.numFeatures
    numIters = args.numIters

    # print maxDepth,minRows


    # load initial data
    data,labels = loadData(inputFile)

    # data = np.array([[2,3,1],[4,5,0],[6,7,1],[8,9,0],[10,11,1]])
    # print data.shape
    # print labels.shape
    classifier = DecisionTree(maxDepth, minRows, numFeatures)

    tree = BoostedClassifier(classifier,numIters,ratio)
    tree.fit(data,labels)
    # exit(0)
    predicted =  tree.predict(data)
    # exit(0)
    # print np.sum(predicted == labels)/ float(labels.shape[0])
    # exit(0)
    print KFoldCrossValidation(tree,data,labels,k=10)

    clf = RandomForestClassifier(max_depth=maxDepth, n_estimators=numClassifiers, bootstrap=True)
    print np.mean(cross_val_score(clf, data, labels, cv=10))



if __name__ == "__main__":
    main(sys.argv[1:])