from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import  VarianceThreshold
from Helper import Helper
from os import listdir
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

class feaSelection(object):

    def getTrainingData(self, start=0, end=2):
        trainingFileList = listdir('data/newdirectionkeys')
        m = len(trainingFileList)
        totalMat =np.empty
        totalLabel = []
        labelCount = 0
        h = Helper()
        for i in range(start,end):
            errorcount = 0.0
            count = 0.0

            fileNameStr = 'data/newdirectionkeys/s_discriptor_' + str(i + 1) + '.key'
            Coordinates_1, trainingMat_1 = h.read_file(fileNameStr)

            fileNameStr = 'data/newdirectionkeys/s_discriptor_' + str(i + 2) + '.key'
            Coordinates_2, trainingMat_2 = h.read_file(fileNameStr)

            fileNameStr = 'data/NewMatchPairs/s_MatchPair_' + str(i + 1) + '.key'
            MatchPair = h.load_pair(fileNameStr)

            eyeMat = np.eye(128,128)
            randomCount = 0
            for j in range(trainingMat_1.shape[0]):
                for k in range(128):
                    eyeMat[(k,k)] = trainingMat_1[j,k]

                smallDot = np.dot(trainingMat_2,eyeMat)

                flabel = np.dot(trainingMat_1[j,:],trainingMat_2.transpose())
                label = np.zeros(flabel.shape)
                for m in range(label.shape[0]):
                    if flabel[m]>=245000:
                       label[m]=1
                       totalLabel.append(1)
                       labelCount += 1
                       if(labelCount==1):
                           totalMat = smallDot[m,:]
                       else:
                           totalMat = np.row_stack((totalMat,smallDot[m,:]))
                    else:
                        if(randomCount==3000 and labelCount>1):
                            totalLabel.append(0)
                            totalMat = np.row_stack((totalMat,smallDot[m,:]))
                            randomCount=0
                        elif labelCount>1:
                            randomCount +=1

        return totalMat, totalLabel

    @staticmethod
    def RFresult(totalMat, totalLabel):
        forest = RandomForestClassifier()
        forest.fit(totalMat,totalLabel)
        importances =   forest.feature_importances_

        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        print("Feature ranking:")

        for f in range(totalMat.shape[1]):
            print("%d. RandomForest -> feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


        # Plot the feature importances of the forest
        '''
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(totalMat.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(totalMat.shape[1]), indices)
        plt.xlim([-1, totalMat.shape[1]])
        plt.show()
        '''
        return indices, importances

    @staticmethod
    def LSresult(totalMat, totalLabel):
        lasso = Lasso(alpha=30, max_iter=10000)
        lasso.fit(totalMat, totalLabel)
        importances = lasso.coef_
        indices = np.argsort(importances)[::-1]


        print("Feature ranking:")

        for f in range(totalMat.shape[1]):
            print("%d. Lasso -> feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        return indices, importances

if __name__ == "__main__":
    f = feaSelection()
    totalMat, totalLabel = f.getTrainingData()
    f.RFresult(totalMat,totalLabel)
    f.LSresult(totalMat, totalLabel)
