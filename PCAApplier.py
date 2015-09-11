# -*-coding:utf-8-*-

import sklearn.decomposition


class PCAApplier(object):

    def __init__(self, contribution):
        self.contribution = contribution

    def applyPCA(self, inputs_data):
        if(len(inputs_data)) == 0:
            return None
        # if 0 < n_components < 1, means contribution
        pca = sklearn.decomposition.PCA(n_components=self.contribution)
        pcaed_data = pca.fit_transform(inputs_data)
        return pcaed_data
