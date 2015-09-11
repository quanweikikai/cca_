#!-*-coding:utf-8-*-
import numpy as np
import copy
from itertools import izip
# from MovieMaker import MovieMaker

"""
命名規則
frames_embededVector = [embeddedVector(t=0), embeddedVector(t=1), ...]
"""


def RemoveMissingValue(items):
    numbers = []
    for line in items[6:]:
        templine = []
        for item in line:
            try:
                templine.append(float(item))
            except:
                return numbers
        numbers.append(templine)
    return numbers


def normalize(data):  # 横位置列をそれぞれ正規化する
    return (data - np.average(data)) / np.std(data)


def Combine3FramesEmbeddedVec(frames_vector1, frames_vector2, frames_vector3):
    return [np.r_[np.r_[vector1, vector2], vector3] for vector1, vector2, vector3 in izip(frames_vector1, frames_vector2, frames_vector3)]


def Combine2FramesEmbeddedVec(frames_vector1, frames_vector2):
    return [np.r_[vector1, vector2] for vector1, vector2 in izip(frames_vector1, frames_vector2)]


def SplitEmbeddedVec(frames_labels, frames_embeddedVec):
    numLabels = max(frames_labels) + 1
    labels_frames_embeddedVec = [[] for i in range(numLabels)]
    for frame_i in range(len(frames_labels)):
        labels_frames_embeddedVec[frames_labels[frame_i]].append(
            frames_embeddedVec[frame_i])
    return labels_frames_embeddedVec


class MotionCaptureDataLoader(object):

    def __init__(self, filename):
        self.filename = filename  # 読み込むファイルは行末の無駄なtabを削除している必要がある 要改良

    def Load(self):
        items = []
        for i, line in enumerate(open(self.filename, "r")):
            line = line[:-2].split("\t")[:23]  # frame + time + 7関節のxyzで23
            if len(line) != 23 and i > 5:  # 最初の方はタイトルとかあるので、短くても気にしない
                continue  # データの欠損点があったらそこまでのデータしか使わない
            items.append(line)

        temp = np.array(RemoveMissingValue(items))
        temp = np.delete(temp, 1, 1)  # frame と 秒が書いてある行を削n除する
        self.frames_positions = np.array(
            np.delete(temp, 0, 1))  # TODO もっと綺麗にかけるはず
        self.convertToRelativePosition()
        frames_speed = np.diff(self.frames_positions, axis=0)
        self.frames_featureVec = np.hstack(
            [self.frames_positions[1:], frames_speed])  # 正規化したほうがいいかも？

    def getNumFrames(self):
        return len(self.frames_featureVec)

    def convertToRelativePosition(self):
        # 中心からの相対位置 これがないと 投げというより体の全体的な移動が反映されるので良くないかなと思ってやる
        frames_centerPos = copy.deepcopy(
            self.frames_positions[:, 3 * 3:3 * 3 + 3])  # centerが3番目 × xyz
        for frame_i, positions in enumerate(self.frames_positions):
            for position_i, j in enumerate(positions):
                self.frames_positions[frame_i][
                    position_i] -= frames_centerPos[frame_i][position_i % 3]

    def adjustNumFrames(self, numFrames):
        self.frames_featureVec = np.delete(
            self.frames_featureVec, np.s_[numFrames:], 0)

    def generateEmbeddedVec(self, embeddedTime, delayTime, numDeleteHeadFrames=0, numSamplingFrames=1):
        frames = []
        frames_embeddedVector = []
        for frame_i in range(embeddedTime + delayTime - 1, len(self.frames_featureVec)):
            temp_frames = []
            tempEmbeddedVector = []
            for embedded_i in range(0, embeddedTime, numSamplingFrames):
                temp_frames.append(frame_i - delayTime - embeddedTime + 1 + embedded_i)
                tempEmbeddedVector.append(self.frames_featureVec[frame_i - delayTime - embeddedTime + 1 + embedded_i])
            tempEmbeddedVector = np.array(tempEmbeddedVector).flatten()
            frames_embeddedVector.append(tempEmbeddedVector)
            frames.append(temp_frames)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.9)
        frames_embeddedVector = pca.fit_transform(frames_embeddedVector)
#        return frames[numDeleteHeadFrames:]
        return frames_embeddedVector[numDeleteHeadFrames:]

    def GenerateDataForMovie(self, numDeleteHeadFrames):
        return np.array(self.frames_positions[numDeleteHeadFrames:])


#    def GenerateLabeledSplitedDataForMovie(self, labels_framesMat, embedded_time, numSamplingFrames):
#        labels_frames_joints_positions = [[] for i in labels_framesMat]
#        for label_i, frames_in_label in enumerate(labels_framesMat):
#            temp_frames = []
#            for frame_in_label in frames_in_label:
#                for frame in range(0, embedded_time, numSamplingFrames):
#                    temp_frames.append(frame_in_label + frame)
#                labels_frames_joints_positions[label_i] = self.frames_positions[temp_frames] #7関節
#        return np.array(labels_frames_joints_positions)
#


if __name__ == '__main__':
    from optparse import OptionParser
    from singleThread_PPCCAEMalgorithmn import PCCAEMalgorithmn
#    from PPCCAEMalgorithmn import PCCAEMalgorithmn

    usage = "clustering with granger causality Y->X. usage: %prog [options] keyword"
    parser = OptionParser(usage)
    parser.add_option(
        "-x", "--file_x",
        default="./data/140910051-person1.trc",
        dest="datafileX",
        help="input data file X")

    parser.add_option(
        "-y", "--file_y",
        default="./data/140910051-person2.trc",
        dest="datafileY",
        help="input data file Y")

    parser.add_option(
        "-e", "--embedded_frame",
        default=20,
        dest="embeddedTime",
        type="int",
        help="create embedded vector with embedded time. numEmbeddedFrames = embeddedFrames/numSamplingFrames")

    parser.add_option(
        "-s", "--sampling_frame",
        default=1,
        dest="numSamplingFrames",
        type="int",
        help="create embedded vector with sampling frames, but all data will be used for embedded vecs")

    parser.add_option(
        "-d", "--delay_frame",
        default="1",
        dest="delayTime",
        type="int",
        help="delay time about embedded vector")

    parser.add_option(
        "-k", "--num_cluster",
        default="5",
        dest="numCluster",
        type="int",
        help="number of clusters")

    (options, args) = parser.parse_args()
    numSamplingFrames = options.numSamplingFrames
    embeddedTime = options.embeddedTime
    delayTime = options.delayTime
    numCluster = options.numCluster
    datafileX = options.datafileX
    datafileY = options.datafileY

    print "causality:" + datafileY + "->" + datafileX + "  embeddedtime %d, delaytime %d, samplingtime %d, numcluster %d" % (embeddedTime, delayTime, numSamplingFrames, numCluster)

    ##########
    loaderX = MotionCaptureDataLoader(datafileX)
    loaderY = MotionCaptureDataLoader(datafileY)

    loaderX.Load()
    loaderY.Load()

    minFrames = min(loaderX.getNumFrames(), loaderY.getNumFrames())
    loaderX.adjustNumFrames(minFrames)
    loaderY.adjustNumFrames(minFrames)

    frames_embeddedVecXt = loaderX.generateEmbeddedVec(1, 0, numDeleteHeadFrames=embeddedTime + delayTime - 1)
    frames_embeddedVecXtkm = loaderX.generateEmbeddedVec(embeddedTime, delayTime, numSamplingFrames=numSamplingFrames)
    frames_embeddedVecYtkm = loaderY.generateEmbeddedVec(embeddedTime, delayTime, numSamplingFrames=numSamplingFrames)

    loaderX = None
    loaderY = None

    data = [[np.matrix(embeddedVecXt).T, np.matrix(embeddedVecYtkm).T, np.matrix(embeddedVecXtkm).T]
            for i, [embeddedVecXt, embeddedVecYtkm, embeddedVecXtkm] in enumerate(izip(frames_embeddedVecXt, frames_embeddedVecYtkm, frames_embeddedVecXtkm))]

    print "num data %d" % len(data)

    EM = PCCAEMalgorithmn(data, numCluster)
#    serial_results = EM.calcUntilNoChangeClustering()
    serial_results = EM.calcSomeStep(100)
    print serial_results

    labels_frames = [[] for i in range(numCluster)]
    for i, label in enumerate(serial_results[-1]):
        labels_frames[label].append(i)

    import time
    import cPickle as pickle

    pickle_dic = {"datafileX": datafileX,
                  "datafileY": datafileY,
                  "serial_results": serial_results,
                  "embeddedTime": embeddedTime,
                  "delayTime": delayTime,
                  "numSamplingFrames": numSamplingFrames,
                  "numCluster": numCluster,
                  "convergence_params": EM.getParamsDictionary(),
                  "eigen_vals": EM.cluster_eigen_vals,
                  "eigen_vecs": EM.cluster_eigen_vecs}
# "data_for_movie_x":loaderX.GenerateLabeledSplitedDataForMovie(labels_frames, embeddedTime, numSamplingFrames),
# "data_for_movie_y":loaderY.GenerateLabeledSplitedDataForMovie(labels_frames, embeddedTime, numSamplingFrames),
# "data":data,


#    filename = "/data1/keisuke.kawano/results/motion_cap/" + (datafileX + datafileY).replace(".trc","_").replace("./data/","")  +"%dclusters_%dembeddedTime_%ddelayTime_%dnumSumpingFrames_"%(numCluster, embeddedTime, delayTime, numSamplingFrames) + str(time.strftime("%Y%b%d_%a_%H%M%S", time.strptime(time.ctime())))+".dat"
    filename = "/home/kei/Desktop/" + (datafileX + datafileY).replace(".trc", "_").replace("./data/", "") + "%dclusters_%dembeddedTime_%ddelayTime_%dnumSumpingFrames_" % (
        numCluster, embeddedTime, delayTime, numSamplingFrames) + str(time.strftime("%Y%b%d_%a_%H%M%S", time.strptime(time.ctime()))) + ".dat"
#    filename = "/Users/kawano/Desktop/" + (datafileX + datafileY).replace(".trc","_").replace("./data/","")  +"%dclusters_%dembeddedTime_%ddelayTime_%dnumSumpingFrames_"%(numCluster, embeddedTime, delayTime, numSamplingFrames) + str(time.strftime("%Y%b%d_%a_%H%M%S", time.strptime(time.ctime())))+".dat"
    pickle_file = open(filename, "w")
    pickle.dump(pickle_dic, pickle_file)
    pickle_file.close()
