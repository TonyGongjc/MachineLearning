from numpy import *
import operator
import time
from os import listdir
from sklearn.decomposition import PCA

def classify(input, dataSet, labels, k):
    dataSetSize=dataSet.shape[0]  #get the number of tuples
    diffMat = tile(input,(dataSetSize,1))- dataSet   #create a differential matrix, substract each row of #
                                                        # training data witht test data
    sqDiffMat=diffMat**2          #square
    sqDistance=sqDiffMat.sum(axis=1) # row aggregation
    Distance=sqDistance**0.5      #square root
    sortedDistanceIndices = Distance.argsort() #sort the labels
    classCount={}
    for i in range(k):
        Kneighbors=labels[sortedDistanceIndices[i]]
        classCount[Kneighbors]=classCount.get(Kneighbors,0)+1 #get they key from Kneighbor
    sortedClassCount=sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def img2vector(filename):
    returnVect=zeros((1,1024))
    fr= open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j]) #read the file and convert it to a 1024 vector
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

def KNNTest(k):
    errorRate=[]
    PcaActive = 1
    if (PcaActive==0):
        trainingLabels = []
        trainingFileList = listdir('digits/trainingDigits')  # read all the files under the directory
        m = len(trainingFileList)
        t1 = time.time()
        trainingMat = zeros((m, 1024))
        for i in range(m):
            fileNameStr = trainingFileList[i]
            fileStr = fileNameStr.split('.')[0]  # split the file name
            classNumStr = int(fileStr.split('_')[0])
            trainingLabels.append(classNumStr)  # get the training labels
            trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)

        testFileList = listdir('digits/trainingDigits')
        errorCount = 0.0
        TestLength = len(testFileList)

    else:
        trainingMat, testingMat, trainingLabels, trainingFileList, testingFileList = data_preparation()
        TestLength = len(testingFileList)
        n = len(testingFileList)
        m = len(trainingFileList)
        newTrainingData, newTestData = PCAtraining(trainingMat, testingMat, 500, m)
        t1 = time.time()

    for test in range (k):
        errorCount = 0.0
        for i in range(TestLength):
            if PcaActive==0:
                fileNameStr=testingFileList[i]
                fileStr = fileNameStr.split('.')[0]
                classNumStr=int(fileStr.split('_')[0])
                TestingVector=img2vector('digits/trainingDigits/%s' % fileNameStr)
                result=classify(TestingVector, trainingMat, trainingLabels, test+1)
            else:
                fileNameStr = testingFileList[i]
                fileStr = fileNameStr.split('.')[0]
                classNumStr = int(fileStr.split('_')[0])
                result=classify(newTestData[i].reshape(1,-1),newTrainingData, trainingLabels,test+1)
            print("the classification result is: %d, the correct class is: %d" % (result, classNumStr))
            if(result != classNumStr):
                errorCount +=1.0
        print("\n total number of test is: %d" % TestLength)
        print("\n total number of errors is: %d" % errorCount)
        print("\n the error rate is: %f" % (errorCount/TestLength))
        t2 = time.time()
        print("\n Cost time: %.2fmin, %.4fs." % ((t2 - t1) // 60, (t2 - t1) % 60))
        errorRate.append(errorCount/TestLength)
    return errorRate



if __name__=="__main__":

    print(KNNTest(10))



