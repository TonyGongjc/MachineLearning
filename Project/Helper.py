from numpy import *

class Helper(object):
    @staticmethod
    def read_file(filename):
        row = sum(1 for line in open(filename))
        returnVect = zeros((row, 128))
        coordinate = zeros((row, 2))
        fr = open(filename)
        for i in range(row):
            lineStr = fr.readline()
            lineStr = lineStr.split()
            for j in range(130):
                if (j > 1):
                    returnVect[i, j - 2] = int(lineStr[j])
                else:
                    coordinate[i, j] = int(lineStr[j])
        return coordinate, returnVect

    @staticmethod
    def load_pair(filename):
        row = sum(1 for line in open(filename))
        MatchPair = zeros((row, 4))
        fr = open(filename)
        for i in range(row):
            lineStr = fr.readline()
            lineStr = lineStr.split()
            for j in range(4):
                MatchPair[i, j] = int(lineStr[j])
        return MatchPair