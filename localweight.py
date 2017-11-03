import numpy as np
import math
from sklearn import preprocessing
import matplotlib.pyplot as plt

class localweight(object):
    def __init__(self, rate = 0.01, lamda=0.001, taw=0.01):
        train = localweight.read_file('bclass/bclass/bclass-train')
        test = localweight.read_file('bclass/bclass/bclass-test')
        self.train_data = np.mat(train[0])
        self.train_col = train[1]
        self.train_row = train[2]
        self.train_label = train[3]
        self.test_data = np.mat(test[0])
        self.test_col = test[1]
        self.test_row = test[2]
        self.test_label = test[3]
        self.rate = rate
        self.weight = np.ones((self.train_col,1))
        self.lamda = lamda
        self.taw = taw
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

    def Weights(self, testPoint):
        w = np.mat(np.eye(self.train_row))
        for j in range(self.train_row):
            diffmat = testPoint - self.train_data[j]
            w[j,j] = np.exp(diffmat*diffmat.transpose()/(-2.0*self.taw**2))
        return w

    def newton(self, iternum , local_weight):
        X=self.train_data.transpose()
        Y=np.mat(self.train_label).transpose()
        row=self.train_row
        col=self.train_col
        beta=np.zeros((col,1))
        A = np.eye(row)
        z = np.ones((row,1))
        p = np.ones((row,1))
        for j in range(iternum):
            for i in range(row):
                x=X[:,i]
                dotProduct = x.transpose()*beta
                h= self.sigmoid(dotProduct)
                aii = h*(1-h)
                z[i] = dotProduct+((1-self.sigmoid(dotProduct)))/aii
                A[i,i] = aii
                p[i]=h
            #first_order = X*local_weight *(Y-p)-2*self.lamda*beta
            #second_order = -X * A*local_weight * X.transpose() - self.lamda * np.eye(col)
            first_order = X  * (Y - p) - 2 * self.lamda * beta
            second_order = -X * A  * X.transpose() - self.lamda * np.eye(col)
            beta = beta - np.linalg.pinv(second_order)*first_order
            self.weight=beta
            self.check_train()
            self.check_test()

    def check_train2(self):
        dataMatrix = np.mat(self.train_data)
        errorcount = 0.0

        for i in range(self.train_row):
            w = localweight.Weights(self, dataMatrix[i])
            self.newton(10,w)
            dotProduct = sum(dataMatrix[i] * self.weight)
            h = self.sigmoid(dotProduct[0,0])
            if h>=0.5:
                result = 1
            else:
                result = 0
            if(result != self.train_label[i]):
                errorcount +=1.0
            print("%d iteration" % (i + 1))
            print("%d errors" % errorcount)
        print("\n total number of train is: %d" % self.train_row)
        print("\n total number of train error is: %d" % errorcount)
        print("\n the train error rate is: %f" % (errorcount/self.train_row))

    def check_test2(self):
        dataMatrix = self.test_data
        errorcount = 0.0
        for i in range(self.test_row):
            w = localweight.Weights(self, dataMatrix[i])
            localweight.newton(self, 10, w)
            dotProduct = sum(dataMatrix[i] * self.weight)
            h = self.sigmoid(dotProduct[0,0])
            if h>=0.5:
                result = 1
            else:
                result = 0
            if(result != self.test_label[i]):
                errorcount +=1.0
            print("%d iteration" % (i+1))
            print("%d errors" % errorcount)

        print("\n total number of test is: %d" % self.test_row)
        print("\n total number of test error is: %d" % errorcount)
        print("\n the test error rate is: %f" % (errorcount/self.test_row))

    def check_train(self):
        dataMatrix = np.mat(self.train_data)
        errorcount = 0.0
        for i in range(self.train_row):
            dotProduct = sum(dataMatrix[i] * self.weight)
            h = self.sigmoid(dotProduct[0, 0])
            if h >= 0.5:
                result = 1
            else:
                result = 0
            if (result != self.train_label[i]):
                errorcount += 1.0
        self.train_error.append(errorcount / self.train_row)
        print("\n total number of train is: %d" % self.train_row)
        print("\n total number of train error is: %d" % errorcount)
        print("\n the train error rate is: %f" % (errorcount / self.train_row))

    def check_test(self):
        dataMatrix = self.test_data
        errorcount = 0.0
        for i in range(self.test_row):
            dotProduct = sum(dataMatrix[i] * self.weight)
            h = self.sigmoid(dotProduct[0, 0])
            if h >= 0.5:
                result = 1
            else:
                result = 0
            if (result != self.test_label[i]):
                errorcount += 1.0
        self.test_error.append(errorcount / self.test_row)
        print("\n total number of test is: %d" % self.test_row)
        print("\n total number of test error is: %d" % errorcount)
        print("\n the test error rate is: %f" % (errorcount/self.test_row))
if __name__ == "__main__":
    GL =localweight(taw=5)
    print(GL.test_data)
    '''
    weight=GL.Weights(GL.test_data[0])
    w=[]
    for i in range(GL.test_row):
        print(weight[i,i])
        w.append(weight[i,i])
    print(max(w))
   
    GL.newton(100,1)
    GL.check_train()
    GL.check_test()
    print(min(GL.train_error), min(GL.test_error))
    plt.plot(GL.train_error[0:30], 'r--', GL.test_error[0:30], 'b--')
    plt.show()
    '''