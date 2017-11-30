import numpy as np
from pandas import DataFrame
# from sklearn.metrics.pairwise import euclidean_distances
import os
import sys
import getopt
import copy
import argparse
import simplejson
import copy
# from sklearn.model_selection import cross_val_score
# from sklearn.tree import DecisionTreeClassifier

def loadData(filePath):
    # function to load data from file

    # if file doesnt exist then return None
    if not os.path.exists(filePath):
        return None

    # load data from the file
    df = DataFrame.from_csv(filePath, sep='\s+', header=None, index_col=None)
    # print df

    obj_cols = df.select_dtypes(include=['object']).columns.values.tolist()

    for col in obj_cols:
        df[col] = df[col].astype('category')

    obj_cols = df.select_dtypes(['category']).columns

    df[obj_cols] = df[obj_cols].apply(lambda x: x.cat.codes)

    data = df.values


    # extract labels and data
    labels = data[:,-1]
    data = data[:,:-1]

    return (data,labels)


class DecisionTree(object):

    def __init__(self, maxDepth=1, minRows=1):
        self.maxDepth = maxDepth
        self.minRows = minRows
        self.root = None

    def fit(self, data, labels):
        self.data = data
        self.labels = labels
        self.data = np.hstack((data,labels.reshape(-1,1)))
        # print self.data.shape
        self.root = self.createTree(self.data)

        # print simplejson.dumps(self.root)

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



        for index in xrange(data.shape[1] - 1):
            values = set(data[:,index].reshape(1,-1).tolist()[0])
            for value in values:
                # print index,value,values
                left = data[data[:, index] < value]
                right = data[data[:, index] >= value]
                
                if left.shape[0] == 0 or right.shape[0] == 0:
                    continue
                giniLeft = self.getGini(left, set(left[:,-1].reshape(1,-1).tolist()[0]))
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
            root['index'] = index
            root['value'] = value
            root['left'] = self.createTree(left,depth+1)
            root['right'] = self.createTree(right,depth+1)
            root['gain'] = gain
            del root['data']
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

    np.random.shuffle(data)

    chunks = np.array_split(data, k)

    predicted = np.zeros((len(chunks),1))

    for i in xrange(k):
        temp = copy.copy(chunks)
        test = temp[i]
        del temp[i]
        train = np.vstack(temp)
        # print train.shape,test.shape
        classifier.fit(train[:,:-1],train[:,-1])
        pred = classifier.predict(test[:,:-1])
        predicted[i] = np.sum(pred == test[:,-1])/ float(pred.shape[0])
        # print predicted[i]

    return np.mean(predicted)


def main(argv):
    # arguments
    parser = argparse.ArgumentParser(description='Hierarchial Agglomerative Clustering')

    # optional arguments
    parser.add_argument('-d', '--maxDepth', help='Maximum Depth of Decision Tree', type=int, default=10)
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
    data,labels = loadData(inputFile)

    # data = np.array([[2,3,1],[4,5,0],[6,7,1],[8,9,0],[10,11,1]])
    # print data.shape
    # print labels.shape

    tree = DecisionTree(maxDepth,minRows)
    tree.fit(data,labels)
    predicted =  tree.predict(data)
    # print np.sum(predicted == labels)/ float(labels.shape[0])
    # exit(0)
    print KFoldCrossValidation(tree,data,labels,k=10)

    # clf = DecisionTreeClassifier(max_depth=maxDepth)
    # print np.mean(cross_val_score(clf, data, labels, cv=10))



if __name__ == "__main__":
    main(sys.argv[1:])
