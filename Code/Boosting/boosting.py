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
    def __init__(self, classifier, numClassifiers=10, numIters=100, ratio=0.632, learning_rate=1.0):
        self.classifier = classifier
        self.numClassifiers = numClassifiers
        self.numIters = numIters
        self.ratio = ratio
        self.importance = [0]*numIters
        self.classifiers = [None]*numIters
        self.learning_rate = learning_rate
        self.counter = 0

    def createSubsets(self, data, weights):
        length = data.shape[0]
        ratio = self.ratio

        dataSets = [None]*self.numClassifiers

        originalDataLength = int(length*ratio)
        # print weights.reshape(1,-1).tolist()
        # print length

        for i in xrange(self.numClassifiers):
            randomIndices = np.random.choice(length, originalDataLength, p=weights.reshape(1,-1).tolist()[0], replace=False)
            dataSets[i] = data[randomIndices]
            randomDuplicating = np.random.choice(dataSets[i].shape[0], length - originalDataLength, replace=True)
            # print dataSets[i].shape
            dups = dataSets[i][randomDuplicating]
            # print dups.shape
            dataSets[i] = np.vstack((dataSets[i], dups))
            # print dataSets[i].shape[0]


        return dataSets

    def fit(self, data, labels):
        # self.data = data
        # self.labels = labels
        self.data = np.hstack((data,labels.reshape(-1,1)))
        numIters = self.numIters
        learning_rate = self.learning_rate
        # print self.data.shape
        # self.dataSets = self.createSubsets(self.data)
        # print len(self.dataSets)

        # print [i.shape for i in self.dataSets]

        weights = np.ones((data.shape[0],1)) / data.shape[0]
        # print np.sum(weights)

        for i in xrange(numIters):
            classifier,error,importance,missed = self.getBestClassifier(weights)
            if error > 0.5:
                self.importance = [0]*self.numClassifiers
                self.classifiers = [None]*self.numClassifiers
                self.counter = 0
            else:
                self.importance[i] = importance
                self.classifiers[i] = classifier
                self.counter = i
                if error < 0.000001:

                    break

                # predictions = classifier.predict(data)
                missed = missed * 2 - 1
                # predictions = predictions * 2 - 1
                # sameOrNot = labels * predictions
                # sameOrNot = (sameOrNot * 2 - 1).reshape(-1,1)
                # print missed.shape,'sameshape'


                # print sameOrNot
                # print np.exp(-importance*labels*predictions).shape
                # print weights.shape
                # exit(0)
                # print importance,'importance'
                weights = weights * np.exp(-importance * missed)
                weights = weights / np.sum(weights)
                # print np.sum(weights),'sum weights'
        # print self.importance[:self.counter+1]
        # print self.classifiers[:self.counter+1]




    def getBestClassifier(self,weights):
        bestClassifier = None
        bestAccuracy = None
        bestPreds = None

        numClassifiers = self.numClassifiers

        dataSets = self.createSubsets(self.data,weights)

        for i in xrange(numClassifiers):
            tempClassifier = self.classifier

            dataSet =  dataSets[i]

            # print dataSet[i].shape, self.data.shape

            tempClassifier.fit(dataSet[:,:-1],dataSet[:,-1])

            preds = tempClassifier.predict(self.data[:,:-1])

            tempAcc = np.sum(preds == self.data[:,-1])/ float(self.data.shape[0])

            # print tempAcc

            if bestAccuracy is None or tempAcc > bestAccuracy:
                bestAccuracy = tempAcc
                bestClassifier = tempClassifier
                bestPreds = preds

        # print weights.shape,bestPreds.shape,self.data[:,-1].shape
        missed = (bestPreds != self.data[:,-1]).reshape(-1,1)
        # missed = missed * 2 - 1

        error = np.dot(weights.T,missed) / np.sum(weights)
        # print error,'error'

        importance = 0.5 * np.log((1-error)/error)

        return bestClassifier,error,importance,missed



    def getMostCount(self,data1):
        # print 'data = ',data1
        label = np.sign(np.sum(data1))

        if label < 0:
            label = 0


        return label

    def predict(self, data):
        labelArr = [self.importance[i] * (self.classifiers[i].predict(data) * 2 -1) for i in xrange(self.counter+1)]

        labels = np.apply_along_axis(self.getMostCount, 0, labelArr).reshape(-1,1)
        # print labels.shape
        # print labels
        return labels




def main(argv):
    # arguments
    parser = argparse.ArgumentParser(description='Random Forest Classifier')

    # optional arguments
    parser.add_argument('-d', '--maxDepth', help='Maximum Depth of Decision Tree', type=int, default=10000)
    parser.add_argument('-r', '--minRows', help='Minimum Rows required to split', type=int, default=1)
    parser.add_argument('-n', '--numClassifiers', help='Number of Classifiers', type=int, default=10)
    parser.add_argument('-x', '--ratio', help='Ratio of data for bagging', type=float, default=1-1/np.e)
    parser.add_argument('-m', '--numIters', help='Number of iterations', type=int, default=50)
    parser.add_argument('-f', '--numFeatures', help='Ratio of number of features for decision tree', type=float, default=0.2)
    parser.add_argument('-l', '--learningRate', help='Ratio of number of features for decision tree', type=float, default=1.0)
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

    # data = np.array([[2,3,1],[4,5,0],[6,7,1],[8,9,0],[10,11,1]])
    # print data.shape
    # print labels.shape
    classifier = DecisionTree(maxDepth, minRows, numFeatures)

    tree = BoostedClassifier(classifier,numClassifiers,numIters,ratio,learningRate)
    # tree.fit(data,labels)
    # exit(0)
    # predicted =  tree.predict(data)
    # exit(0)
    # print np.sum(predicted == labels.reshape(-1,1))/ float(labels.shape[0])
    # exit(0)
    accuracy,precision,recall,f1_score = KFoldCrossValidation(tree,data,labels,k=10)
    print '='*40 + 'Average' + '='*33
    print 'Accuracy:\t',accuracy
    print 'Precision:\t',precision
    print 'Recall:\t',recall
    print 'F1-Score:\t',f1_score
    print '='*80

    # clf = RandomForestClassifier(max_depth=maxDepth, n_estimators=numClassifiers, bootstrap=True)
    # print np.mean(cross_val_score(clf, data, labels, cv=10))



if __name__ == "__main__":
    main(sys.argv[1:])