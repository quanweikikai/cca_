#!-*-coding:utf-8-*-

import numpy as np
import pylab as plt


class ClusteringErrorRateCalculator(object):

    def __init__(self, serial_results, correct_cluster_labels, numCluster):
        self.numCluster = numCluster
        self.correct_cluster_labels = correct_cluster_labels
        correct_data_index_in_cluster = []
        correct_data_index_in_cluster = [[] for i in range(numCluster)]
        for i, correct_cluster_label in enumerate(correct_cluster_labels):
            correct_data_index_in_cluster[correct_cluster_label].append(i)

        # 最後の収束結果からクラスタ番号らしきものを逆算する
        cluster_index = []
        for cluster_i in range(numCluster):
            result_need_in_a_cluster = np.array(
                serial_results[-1])[correct_data_index_in_cluster[cluster_i]]
            cluster_index.append(
                int(round(np.average(result_need_in_a_cluster))))
        self.correct_cluster_labels = [
            cluster_index[label] for label in correct_cluster_labels]
        self.numData = len(self.correct_cluster_labels)

        self.serial_error_rate = []
        for time_t, results in enumerate(serial_results):
            self.serial_error_rate.append(
                np.count_nonzero(results - self.correct_cluster_labels))

    def plot(self, filename=None):
        plt.plot(np.array(self.serial_error_rate) / float(self.numData))
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def compareSomeOtherResult(self, other_results):
        # データを普通にkmeansしただけのやつとかと結果を比べるためのメソッド
        # TODO 適当にコピペしただけなので，もうちょっとどうにか上とまとめられないのか考える
        cluster_index = []
        correct_data_index_in_cluster = []
        correct_data_index_in_cluster = [[] for i in range(self.numCluster)]
        for i, correct_cluster_label in enumerate(self.correct_cluster_labels):
            correct_data_index_in_cluster[correct_cluster_label].append(i)

        for cluster_i in range(self.numCluster):
            result_need_in_a_cluster = np.array(
                other_results)[correct_data_index_in_cluster[cluster_i]]
            cluster_index.append(
                int(round(np.average(result_need_in_a_cluster))))
        temp_correct_cluster_labels = [
            cluster_index[label] for label in self.correct_cluster_labels]
        print np.count_nonzero(other_results - temp_correct_cluster_labels)
