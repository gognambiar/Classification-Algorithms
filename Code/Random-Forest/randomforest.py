from decision import loadData,DecisionTree,KFoldCrossValidation
import getopt
import argparse
import os
import sys
import numpy as np

class RandomForest(object):
    def __init__(self, numTrees=100, maxDepth=10, minRows=1, ratio=0.632, numFeatures=0.2):
        self.numTrees = numTrees
        self.maxDepth = maxDepth
        self.minRows = minRows
        self.ratio = ratio
        self.numFeatures = numFeatures

    def createSubsets(self, data):
        # function to create N training datasets by sampling with replacement
        length = data.shape[0]
        ratio = self.ratio

        # initialize datasets as None
        dataSets = [None]*self.numTrees

        # get number of rows to keep from the dataset
        originalDataLength = int(length*ratio)

        # create rest of the rows by duplicating from the originalDataLength
        for i in xrange(self.numTrees):
            randomIndices = np.random.choice(length, originalDataLength, replace=False)
            dataSets[i] = data[randomIndices]
            randomDuplicating = np.random.choice(dataSets[i].shape[0], length - originalDataLength, replace=True)
            dups = dataSets[i][randomDuplicating]
            dataSets[i] = np.vstack((dataSets[i], dups))

        return dataSets

    def fit(self, data, labels, mappings = {}):

        self.mappings = mappings

        # combine data and labels
        self.data = np.hstack((data,labels.reshape(-1,1)))

        # create subsets
        self.dataSets = self.createSubsets(self.data)

        self.createForest()

    def createForest(self):
        # function to create N decision trees using the subset datasets
        numTrees = self.numTrees
        dataSets = self.dataSets
        maxDepth = self.maxDepth
        minRows = self.minRows
        numFeatures = self.numFeatures

        # store root of trees for predicition
        roots = [None]*numTrees

        for i in xrange(numTrees):
            tree = DecisionTree(maxDepth, minRows, numFeatures)
            tree.fit(dataSets[i][:,:-1],dataSets[i][:,-1])
            roots[i] = tree

        self.roots = roots

    def getMostCount(self,data):
        # function to get label from the trees by majority voting
        unique, counts = np.unique(data, return_counts=True)
        count = zip(unique, counts)
        count = sorted(count, key = lambda x: x[1], reverse=True)
        return count[0][0]

    def predict(self, data):
        # function to predict label for data matrix 

        # get predictions
        labelArr = [i.predict(data) for i in self.roots]

        # convert matrix so labels for a data point by N trees are in one row
        labelArr = np.concatenate(list(zip(*labelArr))).reshape(-1,self.numTrees)

        # get majority vote for label for matrix
        labels = np.apply_along_axis(self.getMostCount, 1, labelArr)

        return labels




def main(argv):
    # arguments
    parser = argparse.ArgumentParser(description='Random Forest Classifier')

    # optional arguments
    parser.add_argument('-d', '--maxDepth', help='Maximum Depth of Decision Tree', type=int, default=10000)
    parser.add_argument('-r', '--minRows', help='Minimum Rows required to split', type=int, default=1)
    parser.add_argument('-n', '--numTrees', help='Number of Decision Trees in Forest', type=int, default=7)
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



    tree = RandomForest(numTrees,maxDepth,minRows,ratio,numFeatures)
    # tree.fit(data,labels, mappings)
    # predicted =  tree.predict(data)
    # print np.sum(predicted == labels)/ float(labels.shape[0])
    
    accuracy,precision,recall,f1_score = KFoldCrossValidation(tree,data,labels,k=10)
    print '='*40 + 'Average' + '='*33
    print 'Accuracy:\t',accuracy
    print 'Precision:\t',precision
    print 'Recall:\t',recall
    print 'F1-Score:\t',f1_score
    print '='*80




if __name__ == "__main__":
    main(sys.argv[1:])