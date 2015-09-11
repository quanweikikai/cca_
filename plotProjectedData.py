#!-*-coding:utf-8-*-

import numpy as np
from itertools import izip

def calcCovarianceMatrix(dataX, dataY):
    #data should be frames_some_someMat
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    average_dataX = np.average(dataX.T, axis=1)
    average_dataY = np.average(dataY.T, axis=1)
    covariance_matrix = np.zeros((len(average_dataX),len(average_dataY)))

    for row_dataX, row_dataY in izip(dataX, dataY):
        covariance_matrix += np.outer((row_dataX - average_dataX), (row_dataY - average_dataY))

    covariance_matrix = covariance_matrix/float(len(dataX)-1) #時間-1で割る 定義によっては時間で割ることもある
    return np.matrix(covariance_matrix)

def calcProjectionVector(dataX, dataY):
    import scipy.linalg
    SigmaXX = calcCovarianceMatrix(dataX, dataX)
    SigmaYY = calcCovarianceMatrix(dataY, dataY)
    SigmaXY = calcCovarianceMatrix(dataX, dataY)

    leftMatrix  = SigmaXY * SigmaYY.I * SigmaXY.T
    rightMatrix = SigmaXX

    eigenValues, eigenVectors =  scipy.linalg.eig(leftMatrix, rightMatrix)
    maxEigenValueIndex = np.where(eigenValues == max(eigenValues))
    maxEigenVector = eigenVectors[maxEigenValueIndex]
    return maxEigenVector/np.linalg.norm(maxEigenVector)


def removeThirdVariable(Xt, Ytkm, Xtkm, framesInLabel):
    Xti = np.array(Xt)[framesInLabel]
    Ytkmi = np.array(Ytkm)[framesInLabel]
    Xtkmi = np.array(Xtkm)[framesInLabel]

    SigmaXtiXtkmi   = calcCovarianceMatrix(Xti,   Xtkmi)
    SigmaYtkmiXtkmi = calcCovarianceMatrix(Ytkmi, Xtkmi)
    SigmaXtkmiXtkmi = calcCovarianceMatrix(Xtkmi, Xtkmi)

    #逆行列が計算出来ないことがあるので、正則化する
    #詳しくは Yamashita et al. Causal Flow を参照
    regularized_eta = 0.00001
    AHat = SigmaXtiXtkmi   * (SigmaXtkmiXtkmi + regularized_eta * np.matrix(np.identity(SigmaXtkmiXtkmi.shape[0]))).I
    BHat = SigmaYtkmiXtkmi * (SigmaXtkmiXtkmi + regularized_eta * np.matrix(np.identity(SigmaXtkmiXtkmi.shape[0]))).I

    XtDash   = Xt   - Xtkm * AHat.T
    YtkmDash = Ytkm - Xtkm * BHat.T
    return XtDash, YtkmDash





if __name__ == '__main__':
    import sys
    argvs = sys.argv
    if len(argvs) != 2:
        print "usage : python %s  pickle_filename" % argvs[0]
        exit(-1)
    saveDir =  argvs[1].replace(".txt", "").replace(".dat", "")
    import os
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    import pickle
    pickle_data = pickle.load(open(argvs[1],"r"))

#    labels_joints_data_x = pickle_data["data_for_movie_x"]
#    labels_joints_data_y = pickle_data["data_for_movie_y"]
#    data = pickle_data["data"]
    serial_results    = pickle_data["serial_results"]
    embeddedTime      = pickle_data["embeddedTime"]
    delayTime         = pickle_data["delayTime"]
    numSamplingFrames = pickle_data["numSamplingFrames"]
    numCluster        = pickle_data["numCluster"]
    eigen_vals        = pickle_data["eigen_vals"]
    eigen_vecs        = pickle_data["eigen_vecs"]
    datafileX         = pickle_data["datafileX"]
    datafileY         = pickle_data["datafileY"]
    #############

    from MotionCaptureDataLoader import MotionCaptureDataLoader
    loaderX = MotionCaptureDataLoader(datafileX)
    loaderY = MotionCaptureDataLoader(datafileY)

    loaderX.Load()
    loaderY.Load()

    minFrames = min(loaderX.getNumFrames(), loaderY.getNumFrames())
    loaderX.adjustNumFrames(minFrames)
    loaderY.adjustNumFrames(minFrames)

    ##
    frames_embeddedVecXt   = loaderX.generateEmbeddedVec(           1,         0, numDeleteHeadFrames = embeddedTime+delayTime-1)
    frames_embeddedVecXtkm = loaderX.generateEmbeddedVec(embeddedTime, delayTime, numSamplingFrames = numSamplingFrames)
    frames_embeddedVecYtkm = loaderY.generateEmbeddedVec(embeddedTime, delayTime, numSamplingFrames = numSamplingFrames)

    frames_labels = serial_results[-1]
    labels_frames = [[] for i in range(numCluster)]
    for frame, label in enumerate(frames_labels):
        labels_frames[label].append(frame)

#    from MovieMaker import SplitData
#    labels_frames_embeddedVecXt,labels_frames_embeddedVecXtkm,labels_frames_embeddedVecYtkm = SplitData(data, frames_labels)


    import pylab as plt

    for label_i in range(numCluster):
        XtDash, YtkmDash = removeThirdVariable(frames_embeddedVecXt,
                                               frames_embeddedVecYtkm,
                                               frames_embeddedVecXtkm,
                                               labels_frames[label_i])

        projectionVectorXt   = calcProjectionVector(XtDash[labels_frames[label_i]],
                                                    YtkmDash[labels_frames[label_i]])
        projectionVectorYtkm = calcProjectionVector(YtkmDash[labels_frames[label_i]],
                                                    XtDash[labels_frames[label_i]])

#        from util import normalizeByInLabel
        projectedXt   = np.array(XtDash   * projectionVectorXt.T).flatten()
        projectedYtkm = np.array(YtkmDash * projectionVectorYtkm.T).flatten()
        from scipy import stats
        slope, intercept, r_value, _, _ = stats.linregress(projectedXt[labels_frames[label_i]],
                                                           projectedYtkm[labels_frames[label_i]])

#        plt.scatter(projectedXt, projectedYtkm, alpha=0.2, linewidth=0)
        plt.scatter(projectedXt[labels_frames[label_i]], projectedYtkm[labels_frames[label_i]],
                    alpha=0.2, linewidth=0, c="r")
        x = [min(projectedXt), max(projectedXt)]
        y = [i * slope + intercept for i in x]
        plt.plot(x,y)
        plt.savefig( saveDir + "/projected_cluster_%d.png"%label_i)
        plt.clf()
