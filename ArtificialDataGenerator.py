#!-*-coding:utf-8-*-

from PPCCADataGenerator import PPCCADataGenerator
import numpy as np
import random

# choose the random one 
def weighted_choice(choices, weights):
    norm_weights = np.array(weights) / float(sum(weights))
    r = random.uniform(0.0, 1.0)
    upto = 0
    for i, c in enumerate(choices):
        if upto + norm_weights[i] > r:
            return c, i
        upto += weights[i]
    return c, i


class ArtificialDataGenerator(object):

    def __init__(self, pi, params_files):
        self.data_generators = []
	for n in range(len(params_files[0])):
			tmp_data = [PPCCADataGenerator(params_file[n]) for params_file in params_files]
			self.data_generators.append(tmp_data)
        self.pi = pi
        self.z = []

    def generate(self, num_data):
        self.data = []
        for data_i in range(num_data):
            data_generator, z_n = weighted_choice(self.data_generators, self.pi)
	    #print data_generator
            self.data.append([generator.generate() for generator in data_generator])
            self.z.append(z_n)
        return self.data

    def generateForStan(self, num_data):  # stan に食わせるためのデータを作る 設計があれ
        Y1 = []
        Y2 = []
        X = []
        answer = []
        for data_i in range(num_data):
            data_generator, z_n = weighted_choice(self.data_generators, self.pi)
            y1_i, y2_i, x_i = data_generator.generate()
            Y1.append(list(np.array(y1_i).reshape(-1,)))
            Y2.append(list(np.array(y2_i).reshape(-1,)))
            X.append(list(np.array(x_i).reshape(-1,)))
            answer.append(z_n)
        return Y1, Y2, X, answer

    def get_z(self):
        return self.z

    def get_pcaedData(self, n_components_):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components_)
        inputs_data = []
        for temp_data in self.data:
            temp_inputs = []
            for temp in temp_data:
                temp_inputs += np.array(temp.reshape(-1,)).tolist()[0]
            inputs_data.append(temp_inputs)
        return pca.fit_transform(np.array(inputs_data))

    def get_data(self):
        inputs_data = []
        for temp_data in self.data:
            temp_inputs = []
            for temp in temp_data:
                temp_inputs += np.array(temp.reshape(-1,)).tolist()[0]
            inputs_data.append(temp_inputs)
        return inputs_data

##################
    def plotPCAedData(self, cluster_colors=None):
        cluster_colors = np.array(cluster_colors)

        import pylab as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)

        pcaed_data = self.pcaedData(3)  # 3次元に落とす

        if cluster_colors is None:
            colors = ["r", "g", "b"]
            data_colors = [colors[temp_z] for temp_z in self.z]
            ax.scatter(pcaed_data[:, 0], pcaed_data[:, 1], pcaed_data[:, 2], "o", color=data_colors)
        else:
            ax.scatter(pcaed_data[:, 0], pcaed_data[:, 1], pcaed_data[:, 2], "o", color=cluster_colors)

        plt.show()

    def clusteringDataWithKmeans(self, numK):
        # 比較用に普通にkmeansするメソッド
        import sklearn.cluster
        pcaed_data = self.get_pcaedData(0.9)  # pca_contribution
        kmeans = sklearn.cluster.KMeans(n_clusters=numK)
        return kmeans.fit_predict(pcaed_data)
###############################################


if __name__ == '__main__':
    generator = ArtificialDataGenerator([0.3,       0.4,       0.3],
                                        [["params1", "params2", "params3"],["params2","params1","params3"],["params3","params2","params1"]])
    print generator.generate(100)
