#! -*-coding:utf-8-*-

import numpy as np
from itertools import izip


def applyPCA(data, pcaContribution=0.99):
    import sklearn.decomposition
    pca = sklearn.decomposition.PCA(n_components=pcaContribution)  # if 0 < n_components < 1, means contribution
    return pca.fit_transform(data)


def clusteringKMeans(data, numK, pcaContribution=0.99):
    import sklearn.cluster

    kmeans = sklearn.cluster.KMeans(n_clusters=numK)
    frames_clusterLabels = kmeans.fit_predict(data)
    clusterLabel_frames = []
    for label_i in range(numK):
        clusterLabel_frames.append([i for i, j in enumerate(frames_clusterLabels) if j == label_i])
    return clusterLabel_frames


def combine3FramesEmbeddedVec(frames_vector1, frames_vector2, frames_vector3):
    return [np.r_[np.r_[vector1, vector2], vector3] for vector1, vector2, vector3 in izip(frames_vector1, frames_vector2, frames_vector3)]


def calcCovarianceMatrix(dataX, dataY):
    # data should be frames_some_someMat
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    average_dataX = np.average(dataX.T, axis=1)
    average_dataY = np.average(dataY.T, axis=1)
    covariance_matrix = np.zeros((len(average_dataX), len(average_dataY)))

    for row_dataX, row_dataY in izip(dataX, dataY):
        covariance_matrix += np.outer((row_dataX - average_dataX), (row_dataY - average_dataY))

    covariance_matrix = covariance_matrix / float(len(dataX) - 1)  # 時間-1で割る 定義によっては時間で割ることもある
    return np.matrix(covariance_matrix)


def probabilisticChoice(probabilistic):
    import random
    r = random.uniform(0, 1)
    sump = 0
    for i, p in enumerate(probabilistic):
        sump += p
        if r < sump:
            return i
    return len(probabilistic) - 1


def normalize(data):
    return (data - np.average(data)) / np.std(data)


def normalizeByInLabel(dataInLabel, alldata):
    return (alldata - np.average(dataInLabel)) / np.std(alldata)
