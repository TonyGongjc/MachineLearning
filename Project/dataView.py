from numpy import *
import operator
import time
from os import listdir
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Variance import feaSelection
from Helper import Helper


class Sift(object):
    def __init__(self, THRESH = 250000, options = 1, component=50):
        self.thresh=THRESH
        self.option=options
        self.errorCount=[]
        self.errorRatio=[]
        self.recall=[]
        self.component=component

    def train(self):
        h = Helper()
        trainingFileList = listdir('data/newdirectionkeys')
        m = len(trainingFileList)
        if self.option == 3:
            choosing = Sift.Rftraining(self.component)

        if self.option == 4:
            choosing = Sift.LStraining(self.component)

        for i in range(m-1):
            errorcount = 0.0
            count = 0.0

            fileNameStr = 'data/newdirectionkeys/s_discriptor_' + str(i + 1) + '.key'
            Coordinates_1, trainingMat_1 = h.read_file(fileNameStr)

            fileNameStr = 'data/newdirectionkeys/s_discriptor_' + str(i + 2) + '.key'
            Coordinates_2, trainingMat_2 = h.read_file(fileNameStr)

            fileNameStr = 'data/NewMatchPairs/s_MatchPair_' + str(i + 1) + '.key'
            MatchPair = h.load_pair(fileNameStr)

            if(self.option==2):
                trainingMat_1, trainingMat_2 = self._PCAtraining(trainingMat_1, trainingMat_2, self.component)

            if(self.option==3):
                trainingMat_1 = dot(trainingMat_1,choosing)
                trainingMat_2 = dot(trainingMat_2,choosing)

            if(self.option==4):
                trainingMat_1 = dot(trainingMat_1, choosing)
                trainingMat_2 = dot(trainingMat_2, choosing)

            crossProduct = (dot(trainingMat_1, trainingMat_2.transpose()))
            ratio, maxVector, maxPosition = self._maxP(crossProduct)

            FirstPoint = zeros((1, 2))
            SecondPoint = zeros((1, 2))

            for j in range(maxVector.shape[0]):
                if (maxVector[j, 0] > self.thresh):
                    if (not array_equal(FirstPoint, Coordinates_1[int(maxPosition[j, 2])])):
                        FirstPoint = Coordinates_1[int(maxPosition[j, 2])]
                        SecondPoint = Coordinates_2[int(maxPosition[j, 0])]
                        Combine = [FirstPoint[0], FirstPoint[1], SecondPoint[0], SecondPoint[1]]
                        Combine = array(Combine)
                        errorcount += self._errorCheck(Combine, MatchPair)
                        count += 1
            ErrorRatio = errorcount / MatchPair.shape[0]
            Recall = count/MatchPair.shape[0]
            self.errorCount.append(errorcount)
            self.errorRatio.append(ErrorRatio)
            self.recall.append(Recall)
            print("The Error number for pair ", i + 1, " is ", self.errorCount[i])
            print("The error ratio for pair ", i+1, " is ", self.errorRatio[i])
            print("The recall is ", self.recall[i])



    def _maxP(self,crossProduct):
        '''
        maxPosition returns the coordinates of the large dot product value
        [1,2,3,4]
        1 is the largest value's column position, which is also the second matrix coordinate
        2 is the second largest value's column position.
        3 and 4 have the same value which are the coordinate for first matrix

        maxVector returns the largest and second largest dot product value
         '''
        maxVector = zeros((crossProduct.shape[0], 2))
        maxPosition = zeros((crossProduct.shape[0], 4))
        ratio = zeros(crossProduct.shape[0])
        for i in range(crossProduct.shape[0]):
            maxPosition[i, :2] = crossProduct[i].argsort()[::-1][:2]
            maxPosition[i, 2:4] = i
            maxVector[i, 0] = crossProduct[i, int(maxPosition[i, 0])]
            maxVector[i, 1] = crossProduct[i, int(maxPosition[i, 1])]
            ratio[i] = maxVector[i, 0] / maxVector[i, 1]
        return ratio, maxVector, maxPosition



    def _errorCheck(self,Combine, MatchPair):
        for i in range(MatchPair.shape[0]):
            if (array_equal(Combine, MatchPair[i])):
                return 0
        return 1

    def _PCAtraining(self,Mat_1,Mat_2, components):
        pcaTrain = PCA(n_components=components)
        row_1=Mat_1.shape[0]
        Mat=vstack([Mat_1,Mat_2])
        pcaTrain.fit(Mat)
        newTrainingData = pcaTrain.transform(Mat)
        return newTrainingData[:row_1], newTrainingData[row_1:]

    @staticmethod
    def Rftraining(featureNumber):
        print("Start RandomForest Feature Selection...\n")
        f = feaSelection()
        totalMat, totalLabel = f.getTrainingData(start=15,end=19)
        indices, importances = f.RFresult(totalMat, totalLabel)
        diagonal = zeros((128,128))
        for i in range(featureNumber):
            diagonal[(indices[i],indices[i])] = 1
        print("RandomForest Stop\n")
        return diagonal

    @staticmethod
    def LStraining(featureNumber):
        print("Start LASSO Feature Selection\n")
        f = feaSelection()
        totalMat, totalLabel = f.getTrainingData(end=3)
        indices, importances = f.LSresult(totalMat, totalLabel)
        diagonal = zeros((128, 128))
        for i in range(featureNumber):
            diagonal[(indices[i], indices[i])] = 1
        print("LASSO Stop\n")
        return diagonal

    def show(self):
        plt.figure(0)
        plt.title("ErrorRatio")
        plt.xlabel("Image No.")
        plt.plot(self.errorRatio)
        plt.ylim((0,0.5))
   #     plt.xticks(linspace(1,113,28))
        plt.show()
        plt.figure(1)
        plt.title("RecallRate")
        plt.xlabel("Image No.")
        plt.plot(self.recall)
        plt.ylim((0,0.8))
      #  plt.xticks(linspace(1, 113, 28))
        plt.show()


if __name__ == "__main__":
    sift= Sift(THRESH=170000,options=4,component=32)
    sift.train()
    sift.show()


