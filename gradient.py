import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
class gradient(object):
    def __init__(self, rate = 0.01):
        train = gradient.read_file('bclass/bclass/bclass-train')
        test = gradient.read_file('bclass/bclass/bclass-test')
        self.train_data = np.mat(train[0])
        self.train_col = train[1]
        self.train_row = train[2]
        self.train_label = train[3]
        self.test_data = test[0]
        self.test_col = test[1]
        self.test_row = test[2]
        self.test_label = test[3]
        self.rate = rate
        self.weight = np.ones((self.train_col,1))
        self.train_error = []
        self.test_error = []

    def read_file(filename):
        fr = open(filename)
        row = sum(1 for line in open(filename))
        data = []
        label = []
        for i in range(row):
            lineStr = fr.readline()
            lineStr = lineStr.split()
            line_data = []
            for j in range(len(lineStr)):
                t = float(lineStr[j])
                line_data.append(t)
            data.append(line_data[1:len(lineStr)])
            if line_data[0] == -1:
                label.append(0)
            else:  label.append(1)

        return data, len(lineStr)-1, row, label

    def sigmoid(self, inX):
        return 1.0 / (1 + np.exp(-1*inX))


    def update(self, numIter = 150, norm = 0):
        weights = np.ones((self.train_col,1))
        dataMatrix = np.mat(self.train_data)
        if norm == 2:
            dataMatrix = preprocessing.normalize(dataMatrix, norm='l2')
        elif norm == 1:
            dataMatrix = preprocessing.normalize(dataMatrix, norm='l1')
        else:
            pass
        labelMat = np.mat(self.train_label).transpose()
        for j in range(numIter):
           for i in range(self.train_row):
               dotProduct = 0
               for k in range(self.train_col):
                    dotProduct += dataMatrix[i,k]*weights[k]
               h = self.sigmoid(dotProduct)
               error = (labelMat[i] - h)
               weights = weights + (self.rate* error *dataMatrix[i]).transpose()
           self.weight = weights
           self.check_train()
           self.check_test()

    def update_average(self, numIter = 150, norm = 2):
        weights = np.zeros((self.train_col,1))
        dataMatrix = np.mat(self.train_data)
        all_weight=np.ones((self.train_col,1))
        if norm == 2:
            dataMatrix = preprocessing.normalize(dataMatrix, norm='l2')
        elif norm == 1:
            dataMatrix = preprocessing.normalize(dataMatrix, norm='l1')
        else:
            pass
        labelMat = np.mat(self.train_label).transpose()
        for j in range(numIter):
           for i in range(self.train_row):
               dotProduct = 0
               for k in range(self.train_col):
                    dotProduct += dataMatrix[i,k]*weights[k]
               h = self.sigmoid(dotProduct)
               error = labelMat[i] - h
               weights = weights + (self.rate* error *dataMatrix[i]).transpose()
               all_weight=all_weight+weights
           self.weight = all_weight/(numIter*self.train_row)
           self.check_train()
           self.check_test()


    def check_train(self):
        dataMatrix = np.mat(self.train_data)
        errorcount = 0.0
        for i in range(self.train_row):
            dotProduct = sum(dataMatrix[i] * self.weight)
            h = self.sigmoid(dotProduct[0,0])
            if h>=0.5:
                result = 1
            else:
                result = 0
            if(result != self.train_label[i]):
                errorcount +=1.0
        self.train_error.append(errorcount / self.train_row)
        print("\n total number of train is: %d" % self.train_row)
        print("\n total number of train error is: %d" % errorcount)
        print("\n the train error rate is: %f" % (errorcount/self.train_row))

    def check_test(self):
        dataMatrix = np.mat(self.test_data)
        errorcount = 0.0
        for i in range(self.test_row):
            dotProduct = sum(dataMatrix[i] * self.weight)
            h = self.sigmoid(dotProduct[0,0])
            if h>=0.5:
                result = 1
            else:
                result = 0
            if(result != self.test_label[i]):
                errorcount +=1.0
        self.test_error.append(errorcount / self.test_row)
        print("\n total number of test is: %d" % self.test_row)
        print("\n total number of test error is: %d" % errorcount)
        print("\n the test error rate is: %f" % (errorcount/self.test_row))

if __name__ == "__main__":
    GL =gradient()
    GL.update_average(numIter= 500, norm=2)
    print(min(GL.train_error),min(GL.test_error))
    plt.plot(GL.train_error, 'r--', GL.test_error, 'b--')
    plt.show()


