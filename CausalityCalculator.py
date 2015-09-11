# -*-coding:utf-8-*-

from itertools import izip
import numpy as np
import scipy
from PCAApplier import PCAApplier


class CausalityCalculator(object):

    def __init__(self, Xt, Xtkm, Ytkm):
        self.Xt = Xt
        self.Xtkm = Xtkm
        self.Ytkm = Ytkm

    def calcRegularizedGrangerCausality(self, pcaContribution, eta_xtkm, eta_xt, eta_ytkm):
        if (len(self.Xt) == 0):
            print "サンプルがありません"
            return False
#        pca = PCAApplier(pcaContribution)
#        import copy
#        self.Xt = copy.deepcopy(pca.applyPCA(self.Xt))
#        self.Xtkm = copy.deepcopy(pca.applyPCA(self.Xtkm))
#        self.Ytkm = copy.deepcopy(pca.applyPCA(self.Ytkm))

        eigen_values, eigen_vecs = self.calcEquation21(eta_xtkm, eta_xt, eta_ytkm)
        print eigen_values[0]
        print eigen_vecs[0]
        eigen_value = max(eigen_values)

        if eigen_value is None:
            print "固有値が計算出来ません"
            return False

        return eigen_value.real, eigen_vecs[0]  # todo temporary
#        return (np.log2(1 / (1 - eigen_value)) * 0.5).real

    def calcEquation21(self, eta_xtkm, eta_xt, eta_ytkm):
        Xt = self.Xt
        Xtkm = self.Xtkm
        Ytkm = self.Ytkm

        sigma_ytkm_xt = self.calcCovarianceMatrix(Ytkm, Xt)
        sigma_ytkm_xtkm = self.calcCovarianceMatrix(Ytkm, Xtkm)
        sigma_xtkm_xtkm = self.calcCovarianceMatrix(Xtkm, Xtkm)
        sigma_xtkm_xt = self.calcCovarianceMatrix(Xtkm, Xt)
        sigma_xt_xt = self.calcCovarianceMatrix(Xt, Xt)
        sigma_xt_xtkm = self.calcCovarianceMatrix(Xt, Xtkm)
        sigma_xt_ytkm = self.calcCovarianceMatrix(Xt, Ytkm)
        sigma_xtkm_ytkm = self.calcCovarianceMatrix(Xtkm, Ytkm)
        sigma_ytkm_ytkm = self.calcCovarianceMatrix(
            Ytkm, Ytkm)  # 対称では？こんなに計算しなくてもいいのでは？

        sigma_ytkm_xt_bar_xtkm_tilde = self.calcEquation20(
            sigma_ytkm_xt, sigma_ytkm_xtkm, sigma_xtkm_xtkm, sigma_xtkm_xt, eta_xtkm)
        sigma_xt_xt_bar_xtkm_tilde = self.calcEquation20(
            sigma_xt_xt, sigma_xt_xtkm, sigma_xtkm_xtkm, sigma_xtkm_xt, eta_xtkm)
        sigma_xt_ytkm_bar_xtkm_tilde = self.calcEquation20(
            sigma_xt_ytkm, sigma_xt_xtkm, sigma_xtkm_xtkm, sigma_xtkm_ytkm, eta_xtkm)
        sigma_ytkm_ytkm_bar_xtkm_tilde = self.calcEquation20(
            sigma_ytkm_ytkm, sigma_ytkm_xtkm, sigma_xtkm_xtkm, sigma_xtkm_ytkm, eta_xtkm)

        left_matrix = sigma_ytkm_xt_bar_xtkm_tilde * \
            (sigma_xt_xt_bar_xtkm_tilde + eta_xt *
             np.identity(len(sigma_xt_xt_bar_xtkm_tilde))).I * sigma_xt_ytkm_bar_xtkm_tilde
        right_matrix = sigma_ytkm_ytkm_bar_xtkm_tilde + eta_ytkm * \
            np.identity(len(sigma_ytkm_ytkm_bar_xtkm_tilde))

        try:
            #            return scipy.linalg.eig(left_matrix, right_matrix)[0]
            return scipy.linalg.eig(left_matrix, right_matrix)
        except:
            return [None]

    def calcEquation20(self, sigma_ytkm_xt, sigma_ytkm_xtkm, sigma_xtkm_xtkm, sigma_xtkm_xt, eta_xtkm):
        sigma_ytkm_xt_bar_xtkm_tilde = sigma_ytkm_xt  \
            - 2 * sigma_ytkm_xtkm * (sigma_xtkm_xtkm + eta_xtkm).I * sigma_xtkm_xt\
            + sigma_ytkm_xtkm * (sigma_xtkm_xtkm + eta_xtkm).I * \
            sigma_xtkm_xtkm * (sigma_xtkm_xtkm + eta_xtkm).I * sigma_xtkm_xt
        # TODO 汚すぎ どうにかする
        return sigma_ytkm_xt_bar_xtkm_tilde

    def calcCovarianceMatrix(self, dataX, dataY):
        # data should be frames_some_someMat
        dataX = np.array(dataX)
        dataY = np.array(dataY)

        average_dataX = np.average(dataX.T, axis=1)
        average_dataY = np.average(dataY.T, axis=1)

        covariance_matrix = np.zeros((len(average_dataX), len(average_dataY)))

        for row_dataX, row_dataY in izip(dataX, dataY):
            # テンソル積
            covariance_matrix += np.outer((row_dataX -
                                           average_dataX), (row_dataY - average_dataY))

        covariance_matrix = covariance_matrix / float(len(dataX))  # 時間で割る
        return np.matrix(covariance_matrix)
