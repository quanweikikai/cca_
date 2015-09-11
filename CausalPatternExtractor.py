#!-*-coding:utf-8-*-

##
# 他クラスタの点を遠ざける
# 誤差の計算方法の変更
# 実データ、高次元データでの検証
# 誤差関数をハフ変換的に
# 正規化のやり方をみなおす
##


import numpy as np
from util import calcCovarianceMatrix
from random import uniform

class CausalPatternExtractor(object):
    def __init__(self, Xt, Ytkm, Xtkm, clusterLabel_frames):
        self.Xt   = np.matrix(Xt)
        self.Ytkm = np.matrix(Ytkm)
        self.Xtkm = np.matrix(Xtkm)
        self.clusterLabel_frames = clusterLabel_frames
        self.step_sumErrors = []
        self.alpha = 0.00   #TODO
        self.epsilon = 0.0 #TODO
        self.numClusters = 3 #TODO 定義するときに自動でやるようにする


    def extract(self, numRoop):
        for i in range(numRoop):
            clusterLabel_frames_error = self._calcClusterLabel_errors()
#            self._reclusteringProbalicistic(clusterLabel_frames_error)
            self._reclustering(clusterLabel_frames_error)
        return self.step_sumErrors, self.frames_clusterLabels


    def _reclustering(self, clusterLabel_frames_error):
        self.step_sumErrors.append(np.array(clusterLabel_frames_error).min(axis=0).sum())  #現状の誤差を記録 本質的でない

        frames_newClusterLabel = np.array(clusterLabel_frames_error).argmin(axis=0)

#        self.epsilon -= 0.001
        self.epsilon = max(self.epsilon, 0)
#        self.alpha  *= 0.999
#        self.epsilon = 0.2

######### 誤差が大きい上位 epsilon を5番目のクラスタに再クラスタリングする
        frames_minError = np.array(clusterLabel_frames_error).min(axis=0)
        borderError = sorted(frames_minError, reverse=True)[int(len(frames_minError) * self.epsilon)]
        for frame_i, cluster_i in enumerate(frames_newClusterLabel):
            if borderError < frames_minError[frame_i]:
                frames_newClusterLabel[frame_i] = 4
#########

######### ランダムに点を別のクラスタに入れる
        for frame_i,_ in enumerate(frames_newClusterLabel):
            if uniform(0,1) < self.alpha:
                from random import randint
                frames_newClusterLabel[frame_i] = randint(0,3)

        clusterLabel_frames = []
        for label_i in range(max(frames_newClusterLabel)+1):
            clusterLabel_frames.append([i for i,j in enumerate(frames_newClusterLabel) if j==label_i])
        self.clusterLabel_frames = clusterLabel_frames
        self.frames_clusterLabels = frames_newClusterLabel
##########





    def _reclusteringProbalicistic(self, clusterLabel_frames_error):
        self.step_sumErrors.append(np.array(clusterLabel_frames_error).min(axis=0).sum())

        frames_clusterLabel_error = np.array(clusterLabel_frames_error).T
        frames_newClusterLabel = []
        for frame_i, clusterLabel_error in enumerate(frames_clusterLabel_error):
            from numpy import exp
            probabilistic = exp(-clusterLabel_error)/(exp(-clusterLabel_error).sum())
            from util import probabilisticChoice
            frames_newClusterLabel.append(probabilisticChoice(probabilistic))
        clusterLabel_frames = []
        for label_i in range(max(frames_newClusterLabel)+1):
            clusterLabel_frames.append([i for i,j in enumerate(frames_newClusterLabel) if j==label_i])

        from copy import deepcopy
        self.clusterLabel_frames = clusterLabel_frames




    def _calcClusterLabel_errors(self):
        clusterLabel_frames_error = []
        for label_i, framesInLabel in enumerate(self.clusterLabel_frames[:self.numClusters]):
            XtDash, YtkmDash = self._removeThirdVariable(framesInLabel)
            projectionVectorXt   = self._calcProjectionVector(XtDash[framesInLabel],
                                                              YtkmDash[framesInLabel])
            projectionVectorYtkm = self._calcProjectionVector(YtkmDash[framesInLabel],
                                                              XtDash[framesInLabel])

            from util import normalizeByInLabel
            projectedXt   = np.array(XtDash   * projectionVectorXt.T).flatten()
            projectedYtkm = np.array(YtkmDash * projectionVectorYtkm.T).flatten()


            from scipy import stats
            slope, intercept, r_value, _, _ = stats.linregress(projectedXt[framesInLabel], projectedYtkm[framesInLabel])



            ################### TODO  L2ノルム以外の誤差を試してみる ############
            error = abs((slope * projectedXt - projectedYtkm + intercept) / (slope**2 + 1)**0.5).flatten()

            clusterLabel_frames_error.append(np.array(error/np.var(error)))
#            clusterLabel_frames_error.append(np.array(error))    #TODO どういう正規化をするべきなのか、ちゃんと考える

        return clusterLabel_frames_error



    def _removeThirdVariable(self, framesInLabel):
        Xti   = self.Xt[framesInLabel]
        Ytkmi = self.Ytkm[framesInLabel]
        Xtkmi = self.Xtkm[framesInLabel]

        SigmaXtiXtkmi   = calcCovarianceMatrix(Xti,   Xtkmi)
        SigmaYtkmiXtkmi = calcCovarianceMatrix(Ytkmi, Xtkmi)
        SigmaXtkmiXtkmi = calcCovarianceMatrix(Xtkmi, Xtkmi)

        #逆行列が計算出来ないことがあるので、正則化する
        #詳しくは Yamashita et al. Causal Flow を参照
        regularized_eta = 0.00001
        AHat = SigmaXtiXtkmi   * (SigmaXtkmiXtkmi + regularized_eta * np.matrix(np.identity(SigmaXtkmiXtkmi.shape[0]))).I
        BHat = SigmaYtkmiXtkmi * (SigmaXtkmiXtkmi + regularized_eta * np.matrix(np.identity(SigmaXtkmiXtkmi.shape[0]))).I

#        AHat = SigmaXtiXtkmi   * SigmaXtkmiXtkmi.I
#        BHat = SigmaYtkmiXtkmi * SigmaXtkmiXtkmi.I

        XtDash   = self.Xt   - self.Xtkm * AHat.T  #TODO 式がちゃんとあっているか確認すべき
        YtkmDash = self.Ytkm - self.Xtkm * BHat.T
        return XtDash, YtkmDash


    def _calcProjectionVector(self, dataX, dataY):
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


if __name__ == '__main__':
    ## Y -> X のcausality
    import sys, time, os
    import pylab as plt



    from random import randint
    frames_clusterLabels = [randint(0,2) for i,_ in enumerate(Xt1)]
    clusterLabel_frames = []
    for label_i in range(max(frames_clusterLabels)+1):
        clusterLabel_frames.append([i for i,j in enumerate(frames_clusterLabels) if j==label_i])

################

    extractor = CausalPatternExtractor(Xt1, Ytkm, Xtkm, clusterLabel_frames)
    step_sumErrors, clusterLabel_frames = extractor.extract(1000)

    color = ["b", "r", "g", "y", "c", "k", "m"]

    for label_i,framesInLabel in enumerate(clusterLabel_frames):
        framesInLabel = np.array(framesInLabel)
        plt.scatter(np.array(np.array(Xt1)[framesInLabel]).flatten(),
                    np.array(np.array(Ytkm)[framesInLabel]).flatten(),
                    linewidth = 0,
                    c = color[label_i],
                    alpha = 0.2,
                    marker = "o")
