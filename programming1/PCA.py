from Bayes import NaiveBayes
from numpy import *
import operator
import time
from os import listdir
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import KNN

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j]) #read the file and convert it to a 1024 vector
    return returnVect

def data_preparation():
    trainingLabels = []
    trainingFileList = listdir('digits/trainingDigits') #read all the files under the directory
    testingFileList = listdir('digits/testDigits')
    m=len(trainingFileList)
    n=len(testingFileList)
    trainingMat = zeros((m+n,1024))
    testingMat = zeros((n,1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0] #split the file name
        classNumStr = int(fileStr.split('_')[0])
        trainingLabels.append(classNumStr)  #get the training labels
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)

    for i in range(n):
        fileNameStr = testingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # split the file name
        classNumStr = int(fileStr.split('_')[0])
        trainingMat[i+m, :] = img2vector('digits/testDigits/%s' % fileNameStr)
        testingMat[i,:] = img2vector('digits/testDigits/%s' % fileNameStr)
    return trainingMat, testingMat, trainingLabels,trainingFileList, testingFileList


def PCAtraining(trainingMat, testingMat, components,m):
    pcaTrain = PCA(n_components= components)
    pcaTrain.fit(trainingMat)
    newTrainingData = pcaTrain.transform(trainingMat[0:m])
    newTestData = pcaTrain.transform(testingMat)
    return newTrainingData, newTestData

def NBC(newTrainingData,trainingLabels, Bnb, Gnb):
    #nb = NaiveBayes(alpha=1,fit_prior=True)
    #Mnb = MultinomialNB()
    #nb.fit(newTrainingData,trainingLabels)
    #Mnb.fit(newTrainingData,trainingLabels)
    Bnb.fit(newTrainingData, trainingLabels)
    Gnb.fit(newTrainingData, trainingLabels)


if __name__ == "__main__":
    t1 = time.time()
    errorCount = errorCount2 = errorCount3 = errorCount4 = 0.0
    trainingMat, testingMat, trainingLabels, trainingFileList, testingFileList = data_preparation()
    n = len(testingFileList)
    m = len(trainingFileList)
    newTrainingData, newTestData = PCAtraining(trainingMat, testingMat, 200, m)
    Bnb = BernoulliNB()
    Gnb = GaussianNB()
    NBC(newTrainingData, trainingLabels, Bnb, Gnb)

    for i in range(n):
       # result1 = nb.predict(newTestData[i])
       # result2 = Mnb.predict(newTestData[i])
        fileNameStr = testingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        result3 = Bnb.predict(newTestData[i].reshape(1, -1))
        result4 = Gnb.predict(newTestData[i].reshape(1, -1))

        #print("the classification_1 result is: %d, the correct class is: %d" % (result1[0], classNumStr))
        #print("the classification_2 result is: %d, the correct class is: %d" % (result2[0], classNumStr))
        print("the classification_3 result is: %d, the correct class is: %d" % (result3[0], classNumStr))
        print("the classification_4 result is: %d, the correct class is: %d" % (result4[0], classNumStr))

        #if (result1[0] != classNumStr):
           # errorCount += 1.0

        #if (result2[0] != classNumStr):
          #  errorCount2 += 1.0
        if (result3[0] != classNumStr):
            errorCount3 += 1.0

        if (result4[0] != classNumStr):
            errorCount4 += 1.0

    for i in range(m):
       # result1 = nb.predict(newTestData[i])
       # result2 = Mnb.predict(newTestData[i])
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        result1 = Bnb.predict(newTrainingData[i].reshape(1, -1))
        result2 = Gnb.predict(newTrainingData[i].reshape(1, -1))
        print("the classification_1 result is: %d, the correct class is: %d" % (result1[0], classNumStr))
        print("the classification_2 result is: %d, the correct class is: %d" % (result2[0], classNumStr))

        if (result1[0] != classNumStr):
            errorCount += 1.0

        if (result2[0] != classNumStr):
            errorCount2 += 1.0
       # if (result3[0] != classNumStr):
         #   errorCount3 += 1.0

       #if (result4[0] != classNumStr):
          #  errorCount4 += 1.0
    print("\n total number of test is: %d" % n)
    print("\n total number of errors is: %d" % errorCount)
    print("\n total number of errors2 is: %d" % errorCount2)
    print("\n total number of errors3 is: %d" % errorCount3)

    print("\n total number of errors4 is: %d" % errorCount4)

    print("\n the error rate is: %f" % (errorCount / n))
    print("\n the error rate2 is: %f" % (errorCount2 / n))
    print("\n the error rate3 is: %f" % (errorCount3 / n))

    print("\n the error rate4 is: %f" % (errorCount4 / n))
    t2 = time.time()
    print("\n Cost time: %.2fmin, %.4fs." % ((t2 - t1) // 60, (t2 - t1) % 60))

