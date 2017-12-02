usage: knn.py [-h] [-n NUM] [-t TRA] [-p TES] [-i INPUT]

KNN

optional arguments:
  -h, --help            show this help message and exit

Required named arguments:
  -n NUM, --num NUM     Number of Neighbors
  -t TRA, --tra TRA     Training data set file name
  -p TES, --tes TES     Testing data set file name
  -i INPUT, --input INPUT
                        Input data file name

This program can run in 2 ways - 

Train and test data sets are given seperately - 
sample command - python knn.py -n 9 -t project3_dataset3_train.txt -p project3_dataset3_test.txt

Single data set is given - 
sample command - python knn.py -i project3_dataset1.txt

If no k is given(num or Number of Neighbors) for this case it computes the best K based on accuracy.
