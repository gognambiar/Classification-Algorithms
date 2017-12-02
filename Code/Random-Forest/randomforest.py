from decision import loadData,DecisionTree,KFoldCrossValidation
import getopt
import argparse
import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class RandomForest(object):
    def __init__(self, numTrees=100, maxDepth=10, minRows=1, ratio=0.632, numFeatures=0.2):
        self.numTrees = numTrees
        self.maxDepth = maxDepth
        self.minRows = minRows
        self.ratio = ratio
        self.numFeatures = numFeatures

    def createSubsets(self, data):
        length = data.shape[0]
        ratio = self.ratio

        dataSets = [None]*self.numTrees

        originalDataLength = int(length*ratio)

        for i in xrange(self.numTrees):
            randomIndices = np.random.choice(length, originalDataLength, replace=False)
            dataSets[i] = data[randomIndices]
            randomDuplicating = np.random.choice(dataSets[i].shape[0], length - originalDataLength, replace=True)
            # print dataSets[i].shape
            dups = dataSets[i][randomDuplicating]
            # print dups.shape
            dataSets[i] = np.vstack((dataSets[i], dups))
            # print dataSets[i].shape[0]


        return dataSets

    def fit(self, data, labels, mappings = {}):
        # self.data = data
        # self.labels = labels
        self.mappings = mappings
        self.data = np.hstack((data,labels.reshape(-1,1)))
        # print self.data.shape
        self.dataSets = self.createSubsets(self.data)
        # print len(self.dataSets)

        # print [i.shape for i in self.dataSets]
        self.createForest()

    def createForest(self):
        numTrees = self.numTrees
        dataSets = self.dataSets
        maxDepth = self.maxDepth
        minRows = self.minRows
        numFeatures = self.numFeatures

        roots = [None]*numTrees

        for i in xrange(numTrees):
            tree = DecisionTree(maxDepth, minRows, numFeatures)
            tree.fit(dataSets[i][:,:-1],dataSets[i][:,-1])
            roots[i] = tree

        self.roots = roots

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
        # print self.roots
        labelArr = [i.predict(data) for i in self.roots]
        # print labelArr
        labelArr = np.concatenate(list(zip(*labelArr))).reshape(-1,self.numTrees)
        # print labelArr.shape
        labels = np.apply_along_axis(self.getMostCount, 1, labelArr)
        return labels




def main(argv):
    # arguments
    parser = argparse.ArgumentParser(description='Random Forest Classifier')

    # optional arguments
    parser.add_argument('-d', '--maxDepth', help='Maximum Depth of Decision Tree', type=int, default=10)
    parser.add_argument('-r', '--minRows', help='Minimum Rows required to split', type=int, default=1)
    parser.add_argument('-n', '--numTrees', help='Number of Decision Trees in Forest', type=int, default=100)
    parser.add_argument('-x', '--ratio', help='Ratio of data for bagging', type=float, default=1-1/np.e)
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
    numTrees = args.numTrees
    ratio = args.ratio
    numFeatures = args.numFeatures

    # print maxDepth,minRows


    # load initial data
    data,labels, mappings = loadData(inputFile)

    # data = np.array([[2,3,1],[4,5,0],[6,7,1],[8,9,0],[10,11,1]])
    # print data.shape
    # print labels.shape

    tree = RandomForest(numTrees,maxDepth,minRows,ratio,numFeatures)
    tree.fit(data,labels, mappings)
    # exit(0)
    predicted =  tree.predict(data)
    # exit(0)
    # print np.sum(predicted == labels)/ float(labels.shape[0])
    # exit(0)
    accuracy,precision,recall,f1_score = KFoldCrossValidation(tree,data,labels,k=10)
    print '='*40 + 'Average' + '='*33
    print 'Accuracy:\t',accuracy
    print 'Precision:\t',precision
    print 'Recall:\t',recall
    print 'F1-Score:\t',f1_score
    print '='*80

    clf = RandomForestClassifier(max_depth=maxDepth, n_estimators=numTrees, bootstrap=True)
    print np.mean(cross_val_score(clf, data, labels, cv=10))



if __name__ == "__main__":
    main(sys.argv[1:])