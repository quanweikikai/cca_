#!-*-coding:utf-8-*-

import sys
sys.path.append('/home/ken/ppcca/params/')

import numpy as np
import pickle

# TODO パラメータの次数があっているのか確認するメソッド


class PPCCADataGenerator(object):

    """
    パラメータを与えてPPCCAでサンプリングを行うクラス
    基本的にはArtificialDataGeneratorから呼ぶことを想定してデータは一セットづつ吐き出す
    """

    def __init__(self, params_module_name,colNum=3):
        self.params = __import__(params_module_name)
        self.dimension_1 = self.params.mu1.shape[0]
        self.dimension_2 = self.params.mu2.shape[0]
        self.dimension_t = min(self.dimension_1, self.dimension_2)
        self.dimension_x = self.params.mean_x.shape[0]
        self.colNum = colNum

        self.params_module_name = params_module_name

    def generate(self):
        t = np.matrix([np.random.normal(0, 1) for i in range(self.dimension_t)]).T
        x = np.matrix([np.random.normal(self.params.mean_x[i], self.params.sdev_x[i]) for i in range(self.dimension_x)]).T

        # matrix から listに変換してる そのままだとmultivariate_normal()に渡せない
        mean_1 = list(np.array(self.params.Wx1 * x + self.params.Wt1 * t + self.params.mu1).reshape(-1,))
        mean_2 = list(np.array(self.params.Wx2 * x + self.params.Wt2 * t + self.params.mu2).reshape(-1,))
        y1 = np.matrix(np.random.multivariate_normal(mean_1, self.params.Psi1)).T
        y2 = np.matrix(np.random.multivariate_normal(mean_2, self.params.Psi2)).T

        data_one = [y1, y2, x]
        return data_one
   
#    def generate(self):
#		self.data=[]
#		for i in range(self.colNum):
#			self.data.append(self.generate_one())
#		return self.data
			

    def writeDataToPickle(self, filename):
        pkl_file = open(filename, "wb")
        pickle.dump(self.data, pkl_file)
        pkl_file.close()

    def loadDataFromPickle(self, filename):
        pkl_file = open(filename, "rb")
        self.data = pickle.load(pkl_file)


if __name__ == '__main__':
    generator = PPCCADataGenerator("params1")
    print len(generator.generate())
    #generator.writeDataToPickle("test.txt")
    #generator.loadDataFromPickle("test.txt")

#FIXME
