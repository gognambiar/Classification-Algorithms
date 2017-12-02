from decision import loadData,DecisionTree,KFoldCrossValidation
import getopt
import argparse
import os
import sys
import numpy as np
from copy import copy

class BoostedClassifier(object):
    def __init__(self, classifier, numClassifiers=10, numIters=100, ratio=0.632, learning_rate=1.0):
        self.classifier = classifier
        self.numClassifiers = numClassifiers
        self.numIters = numIters
        self.ratio = ratio
        self.alpha = [0]*numIters
        self.classifiers = [None]*numIters
        self.learning_rate = learning_rate
        self.counter = 0

    def createSubsets(self, data, weights):
        # function to create N training datasets by sampling with replacement using the weights as probabilities
        length = data.shape[0]
        ratio = self.ratio

        # create N datasets 
        dataSets = [None]*self.numClassifiers

        # get number of rows to keep from the dataset
        originalDataLength = int(length*ratio)

        # create rest of the rows by duplicating from the originalDataLength
        for i in xrange(self.numClassifiers):
            randomIndices = np.random.choice(length, originalDataLength, p=weights.reshape(1,-1).tolist()[0], replace=False)
            dataSets[i] = data[randomIndices]
            randomDuplicating = np.random.choice(dataSets[i].shape[0], length - originalDataLength, replace=True)
            dups = dataSets[i][randomDuplicating]
            dataSets[i] = np.vstack((dataSets[i], dups))

        return dataSets

    def fit(self, data, labels):
        # combine data and labels
        self.data = np.hstack((data,labels.reshape(-1,1)))
        numIters = self.numIters
        learning_rate = self.learning_rate

        # initialize weights 
        weights = np.ones((data.shape[0],1)) / data.shape[0]

        # for each iteration
        for i in xrange(numIters):
            # get best weak leaenrer for the weights
            classifier,error,alpha,missed = self.getBestClassifier(weights)

            # if error unacceptable then restart
            if error > 0.5:
                self.alpha = [0]*self.numClassifiers
                self.classifiers = [None]*self.numClassifiers
                self.counter = 0
            else:
                # store the alpha and classifier for the iteration
                self.alpha[i] = alpha
                self.classifiers[i] = classifier
                self.counter = i
                if error < 0.000000000000001:

                    break

                # convert labels to -1,1 from 0,1
                missed = missed * 2 - 1

                # get new weights
                weights = weights * np.exp(-alpha * missed)
                # normalize so sum is 1
                weights = weights / np.sum(weights)



    def getBestClassifier(self,weights):
        # function to get best classifier for the specified weights
        bestClassifier = None
        bestAccuracy = None
        bestPreds = None

        numClassifiers = self.numClassifiers

        # create subsets of train with replacements using weights as probabilities
        dataSets = self.createSubsets(self.data,weights)

        # for each classifier calculate error, alpha and accuracy
        for i in xrange(numClassifiers):
            tempClassifier = self.classifier

            dataSet =  dataSets[i]

            tempClassifier.fit(dataSet[:,:-1],dataSet[:,-1])

            preds = tempClassifier.predict(self.data[:,:-1])

            tempAcc = np.sum(preds == self.data[:,-1])/ float(self.data.shape[0])


            if bestAccuracy is None or tempAcc > bestAccuracy:
                bestAccuracy = tempAcc
                bestClassifier = tempClassifier
                bestPreds = preds

        missed = (bestPreds != self.data[:,-1]).reshape(-1,1)

        error = np.dot(weights.T,missed) / np.sum(weights)

        alpha = 0.5 * np.log((1-error)/error)

        return bestClassifier,error,alpha,missed



    def getMostCount(self,data1):
        # function to get label from the trees by majority voting
        label = np.sign(np.sum(data1))

        if label < 0:
            label = 0


        return label

    def predict(self, data):
        # function to predict label for data matrix 

        # get predictions by weighing by alpha
        labelArr = [self.alpha[i] * (self.classifiers[i].predict(data) * 2 -1) for i in xrange(self.counter+1)]

        labels = np.apply_along_axis(self.getMostCount, 0, labelArr).reshape(-1,1)

        return labels




def main(argv):
    # arguments
    parser = argparse.ArgumentParser(description='Boosting Decision Tree Classifier')

    # optional arguments
    parser.add_argument('-d', '--maxDepth', help='Maximum Depth of Decision Tree', type=int, default=10000)
    parser.add_argument('-r', '--minRows', help='Minimum Rows required to split', type=int, default=1)
    parser.add_argument('-n', '--numClassifiers', help='Number of Classifiers', type=int, default=7)
    parser.add_argument('-x', '--ratio', help='Ratio of data for bagging', type=float, default=1-1/np.e)
    parser.add_argument('-m', '--numIters', help='Number of iterations', type=int, default=15)
    parser.add_argument('-f', '--numFeatures', help='Ratio of number of features for decision tree', type=float, default=0.2)
    parser.add_argument('-l', '--learningRate', help='Learning Rate for Boosting', type=float, default=1.0)
    # parser.add_argument('-o', '--output', help='Output file to store PCA visualization')

    # required arguments
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input data file name', required=True, type=str)
    
    args = parser.parse_args()
    
    # parse arguments
    inputFile = args.input
    maxDepth = args.maxDepth
    minRows = args.minRows
    numClassifiers = args.numClassifiers
    ratio = args.ratio
    numFeatures = args.numFeatures
    numIters = args.numIters
    learningRate = args.learningRate

    # print maxDepth,minRows


    # load initial data
    data,labels,mappings = loadData(inputFile)


    classifier = DecisionTree(maxDepth, minRows, numFeatures)

    tree = BoostedClassifier(classifier,numClassifiers,numIters,ratio,learningRate)
    # tree.fit(data,labels)
    # predicted =  tree.predict(data)
    # print np.sum(predicted == labels.reshape(-1,1))/ float(labels.shape[0])
    accuracy,precision,recall,f1_score = KFoldCrossValidation(tree,data,labels,k=10)
    print '='*40 + 'Average' + '='*33
    print 'Accuracy:\t',accuracy
    print 'Precision:\t',precision
    print 'Recall:\t',recall
    print 'F1-Score:\t',f1_score
    print '='*80




if __name__ == "__main__":
    main(sys.argv[1:])