from Bayes import NaiveBayes
from numpy import *
import operator
import time
from os import listdir
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

def img2vector(filename):
    returnVect=zeros((1,1024))
    fr= open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j]) #read the file and convert it to a 1024 vector
    return returnVect


trainingLabels = []
trainingFileList= listdir('digits/trainingDigits') #read all the files under the directory
m=len(trainingFileList)

trainingMat=zeros((m,1024))
for i in range(m):
    fileNameStr=trainingFileList[i]
    fileStr = fileNameStr.split('.')[0] #split the file name
    classNumStr=int(fileStr.split('_')[0])
    trainingLabels.append(classNumStr)  #get the training labels
    trainingMat[i,:]=img2vector('digits/trainingDigits/%s' % fileNameStr)

nb = NaiveBayes(alpha=1,fit_prior=True)
Mnb=MultinomialNB()
Bnb=BernoulliNB()
Gnb=GaussianNB()
nb.fit(trainingMat,trainingLabels)
Mnb.fit(trainingMat,trainingLabels)
Bnb.fit(trainingMat, trainingLabels)
Gnb.fit(trainingMat, trainingLabels)
print("Training finished")


testFileList=listdir('digits/trainingDigits')
t1 = time.time()
#errorCount=0.0
errorCount2=0.0
#errorCount3=0.0
#errorCount4=0.0
TestLength = len(testFileList)
for i in range(TestLength):
    fileNameStr=testFileList[i]
    fileStr = fileNameStr.split('.')[0]
    classNumStr=int(fileStr.split('_')[0])
    TestingVector=img2vector('digits/trainingDigits/%s' % fileNameStr)
    result=nb.predict(TestingVector)
    result2=Mnb.predict(TestingVector)
    result3=Bnb.predict(TestingVector)
    result4 = Gnb.predict(TestingVector)
   # print("the classification_1 result is: %d, the correct class is: %d" % (result[0], classNumStr))
    print("the classification_2 result is: %d, the correct class is: %d" % (result2[0], classNumStr))
   # print("the classification_3 result is: %d, the correct class is: %d" % (result3[0], classNumStr))
   # print("the classification_4 result is: %d, the correct class is: %d" % (result4[0], classNumStr))
    #if(result[0] != classNumStr):
   #     errorCount +=1.0
    if(result2[0]!=classNumStr):
        errorCount2 +=1.0
   # if(result3[0] != classNumStr):
    #    errorCount3 += 1.0
    #if (result4[0] != classNumStr):
    #    errorCount4 += 1.0
print("\n total number of test is: %d" % TestLength)
#print("\n total number of errors is: %d" % errorCount)
print("\n total number of errors2 is: %d" % errorCount2)
#print("\n total number of errors3 is: %d" % errorCount3)
#print("\n total number of errors4 is: %d" % errorCount4)
#print("\n the error rate is: %f" % (errorCount/TestLength))
print("\n the error rate2 is: %f" % (errorCount2/TestLength))
#print("\n the error rate3 is: %f" % (errorCount3/TestLength))
#print("\n the error rate4 is: %f" % (errorCount4/TestLength))
t2 = time.time()
print("\n Cost time: %.2fmin, %.4fs." % ((t2 - t1) // 60, (t2 - t1) % 60))
