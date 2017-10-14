from PIL import Image
from numpy import *
import numpy as np
from os import listdir


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
trainingMat=zeros((1,1024))
fileNameStr=trainingFileList[189]
fileStr = fileNameStr.split('.')[0] #split the file name
trainingMat[0,:]=img2vector('digits/trainingDigits/%s' % fileNameStr)
print(trainingMat)

img=Image.new('1',(32,32))
pixels = img.load()

for i in range(img.size[0]):
    for j in range(img.size[1]):
        pixels[j,i]=int(trainingMat[0][32*i+j])

img.show()
img.save('C:\\Users\\Tony Stark\\Desktop\\765 Machine Learning\\digit.jpg')