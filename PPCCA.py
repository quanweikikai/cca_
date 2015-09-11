#!-*-coding:utf-8-*-

from ArtificialDataGenerator import ArtificialDataGenerator
import numpy as np
from numpy import pi, exp
import warnings
import sys
import multiprocessing as mp
import functools
from scipy.linalg import sqrtm
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


# def calcMultivariateNormal(y, mean, covariance, stabilization_item=0):
# return float(1/(np.linalg.det(2*pi*covariance))**0.5 * exp(-0.5* (y-mean).T * np.linalg.inv(covariance) * (y-mean)))
#    #大きい行列を入れると普通のdetはoverflowで落ちる
#    sign,logdet = np.linalg.slogdet(2*pi*covariance) #正定値行列のdetより常に正なのでsignは捨てる
#    det = np.exp(logdet - stabilization_item) #普通に計算するとeの2000乗とかになるので適当な項を引く このあと結局gamma=(割合)にするので等価
#
#    ####TODO ちゃんと確認する
#    return float((1/det)**0.5 * exp(-0.5* (y-mean).T * np.linalg.inv(covariance) * (y-mean)))


def calcMultivariateNormal(y, mean, covariance, covariance_inv, stabilization_item=0):
    # 大きい行列を入れると普通のdetはoverflowで落ちる
    sign, logdet = np.linalg.slogdet(2 * pi * covariance)  # 正定値行列のdetより常に正なのでsignは捨てる
    det = np.exp(logdet - stabilization_item)  # 普通に計算するとeの2000乗とかになるので適当な項を引く このあと結局gamma=(割合)にするので等価
    return float((1 / det)**0.5 * exp(-0.5 * (y - mean).T * covariance_inv * (y - mean)))


def calcLogMultivariateNormal(y, mean, covariance, covariance_inv):
    sign, logdet = np.linalg.slogdet(2 * pi * covariance)
    try:
        return float(-0.5 * logdet - 0.5 * (y - mean).T * covariance_inv * (y - mean))
    except:
        print logdet
        print np.linalg.det(2 * pi * covariance)
        exit(-1)

def calcData_mnk(gamma,gammaSum,data,m,n,k):
	sum_ry=np.matrix(np.zeros(data[0][m].shape))
	for n_ in range(len(data)):
		sum_ry += gamma[n_,k]*data[n_][m]

	return data[n][m]- (sum_ry/gammaSum[0,k])
	


def calcGamma(data_n, K, Wx, mu, C, C_inv, pi):
        weighted_pdf=[]
        #switchStandard=True
	for k in range(K):
		mul=1.0

                
		for m in range(len(data_n)):
			y1 = data_n[m][0]
			y2 = data_n[m][1]
			y = np.r_[y1, y2]
			x = data_n[m][2]
			log_pdfs = [] 
                        #print len(C) 
			mean = Wx[m][k] * x + mu[m][k]
			covariance = C[m][k]
			covariance_inv = C_inv[m][k]
                      #  if switchStandard:
                       #     standard=calcMultivariateNormal(y,mean,covariance,covariance_inv)
                        #    switchStandard=False
			mul*=calcMultivariateNormal(y,mean,covariance,covariance_inv) #/standard
                        #print mul
		weighted_pdf.append(mul)
	try:
                sum_pdf=sum(weighted_pdf)
		#print [float(pdf)/sum_pdf for pdf in weighted_pdf]
		return [float(pdf)/sum_pdf for pdf in weighted_pdf]
	except RuntimeWarning:
		return [1.0 / K for k in range(K)]


class PCCAEMalgorithmn(object):

    def __init__(self, data, K):
        self.data = data
        self.K = K
        #dimension_1 = data[0][0].shape[0]
        #dimension_2 = data[0][1].shape[0]
        #dimension_x = data[0][2].shape[0]
        #dimension_t = min(dimension_1, dimension_2)

        #print dimension_1, dimension_2, dimension_x
        
	self.num_data = len(data)
   

        #print "data dimension %d %d %d" % (dimension_1, dimension_2, dimension_x)

        self.gamma = np.zeros((len(data), self.K))

        #self.pi = [np.random.rand() for i in range(self.K)]
        self.pi = [1.0/self.K for i in range(self.K)]
        #self.pi /= sum(np.array(self.pi))
	''' this is the parameters that all the data share'''
	self.mu=[]
	self.Wx=[]
	self.C=[]
	self.Psi=[]

	for m in range(len(data[0])):
			dimension_1 = data[0][m][0].shape[0]
			dimension_2 = data[0][m][1].shape[0]
			dimension_x = data[0][m][2].shape[0]
			dimension_t = min(dimension_1, dimension_2)
			mu = [np.matrix(np.random.random((dimension_1 + dimension_2, 1))) for i in range(self.K)]
			Wx = [np.matrix(np.random.random((dimension_1 + dimension_2, dimension_x))) for i in range(self.K)]
			C = []
			Psi = []
			Wt_square=[]
			for k in range(self.K):
				Wt_k = np.matrix(np.random.random((dimension_1 + dimension_2, dimension_t)))*50
				Wt_square_k = Wt_k * Wt_k.T
				Wt_square.append(Wt_square_k)

				temp_matrix1_k = np.matrix(np.random.random((dimension_1, dimension_1)))*50
				temp_matrix2_k = np.matrix(np.random.random((dimension_2, dimension_2)))*50
				Psi1_k = temp_matrix1_k.T * temp_matrix1_k
				Psi2_k = temp_matrix2_k.T * temp_matrix2_k
				zero_matrix1 = np.matrix(np.zeros((dimension_1, dimension_2)))
				zero_matrix2 = np.matrix(np.zeros((dimension_2, dimension_1)))
				Psi_k = np.r_[np.c_[Psi1_k, zero_matrix1], np.c_[zero_matrix2, Psi2_k]]
				Psi.append(Psi_k)
				C_k = Wt_square_k + Psi_k
				C.append(C_k)
                        self.mu.append(mu)
                        self.Wx.append(Wx)
			self.C.append(C)
			self.Psi.append(Psi)
			#self.Wt_square(Wt_square)

        self.normalize_matrix_C = np.identity(dimension_1 + dimension_2) * 0.0  # * 1.0e-7 #正則化項
        self.normalize_matrix_Wx = np.identity(dimension_x) * 0.0  # * 1.0e-7 #正則化項
        self.normalize_gamma = 1.0 / float(self.K)  # 正則化項

    def calcOneStep(self):
	dataNum=len(self.data)
        # calc inverse of C
        sys.stdout.write("\r Estep : calc gamma")
        sys.stdout.flush()

	C_inv=[]
	for m in range(len(self.C)):
            C_inv_m=[]
	    for i, C_k in enumerate(self.C[m]):
                self.C[m][i] = C_k + self.normalize_matrix_C
		C_inv_m.append(np.linalg.inv(C_k))
            C_inv.append(C_inv_m)

	gammaList=[calcGamma(data_n, self.K, self.Wx, self.mu, self.C, C_inv, self.pi) for data_n in self.data]
        self.gamma=np.matrix(gammaList)
	M=len(self.data[0])
        
        #sum of gamma
        sum_gamma = np.sum(self.gamma, axis=0)

        # Mstep
	y=[]
	x=[]
	for data_ in self.data:
		y.append( [np.r_[y1, y2] for i, [y1, y2, _] in enumerate(data_)])
		x.append( [x_n for i, [y1, y2, x_n] in enumerate(data_)])
        
	# calc pi
        sys.stdout.write("\r Mstep : calc pi       ")
        sys.stdout.flush()

        self.pi = sum_gamma / len(self.data)



	for m_ in range(len(self.data[0])):
			d1=y[0][m_].shape[0]
			d2=x[0][m_].shape[0]
			##
			# calc mu
			sys.stdout.write("\r Mstep : calc mu      ")
			sys.stdout.flush()
			for k in range(self.K):
				tmp_1=0
				for n in range(dataNum):
					G_mnk=y[n][m_]-self.Wx[m_][k]*x[n][m_]
					tmp_1+=self.gamma[n,k]*G_mnk

				self.mu[m_][k]= tmp_1 / sum_gamma[0,k]
			##
			# calc Wx  
			sys.stdout.write("\r Mstep : calc Wx        ")
			sys.stdout.flush()
			
			for k in range(self.K):
				tmp_2=np.matrix(np.zeros((d1,d2)))
				tmp_3=np.matrix(np.zeros((d2,d2)))
				for n in range(dataNum):
					y_mnk=calcData_mnk(self.gamma,sum_gamma,y,m_,n,k)
					x_mnk=calcData_mnk(self.gamma,sum_gamma,x,m_,n,k)
					tmp_2+=self.gamma[n,k]*y_mnk*(x_mnk.T)
					tmp_3+=self.gamma[n,k]*x_mnk*(x_mnk.T)

				self.Wx[m_][k] = tmp_2 * (tmp_3.I)
				
			# calc Wt
			sys.stdout.write("\r Mstep : calc C")
			sys.stdout.flush()
			
			for k in range(self.K):
				S_mk = np.matrix(np.zeros((d1,d1)))
				for n in range(dataNum):
					y_mnk_=calcData_mnk(self.gamma,sum_gamma,y,m_,n,k)
					x_mnk_=calcData_mnk(self.gamma,sum_gamma,x,m_,n,k)

					S_mk += self.gamma[n,k]*(y_mnk_-self.Wx[m_][k]*x_mnk_)*((y_mnk_-self.Wx[m_][k]*x_mnk_).T)

				S_mk /= sum_gamma[0,k]
				self.C[m_][k] = S_mk

        return self.gamma

    def calcSomeStep(self, num_step):
        serial_results = []
        serial_results.append(self.calcOneStep().argmax(axis=1))
        for i in range(num_step):
            serial_results.append(self.calcOneStep().argmax(axis=1))
            print ""
            print "%d step - updated %d labels" % (len(serial_results), (np.count_nonzero(serial_results[-1] - serial_results[-2])))
            if(np.array_equal(serial_results[-2], serial_results[-1])):
                warnings.resetwarnings()  # 警告の処理を元に戻しておかないとどうでもいい警告で落ちる
                return serial_results

        warnings.resetwarnings()  # 警告の処理を元に戻しておかないとどうでもいい警告で落ちる
        return serial_results

    def calcUntilNoChangeClustering(self):
        serial_results = []
        serial_results.append(self.calcOneStep().argmax(axis=1))
        while(1):
            serial_results.append(self.calcOneStep().argmax(axis=1))
            print ""
            print "%d step - updated %d labels" % (len(serial_results), (np.count_nonzero(serial_results[-1] - serial_results[-2])))

            if(np.array_equal(serial_results[-2], serial_results[-1])):
                warnings.resetwarnings()  # 警告の処理を元に戻しておかないとどうでもいい警告で落ちる
                return serial_results

    def calcUntilNoChangeGamma(self):
        serial_results = []
        serial_gamma = []
        serial_gamma.append(self.calcOneStep().copy())
        serial_results.append(serial_gamma[-1].argmax(axis=1))
        while(1):
            serial_gamma.append(self.calcOneStep().copy())
            serial_results.append(serial_gamma[-1].argmax(axis=1))
            print len(serial_results)
            print serial_results[-1]
            print np.count_nonzero(serial_gamma[-1] - serial_gamma[-2])
            print serial_gamma[-1] - serial_gamma[-2]

            if(np.array_equal(serial_gamma[-2], serial_gamma[-1])):
                return serial_results

    def getParamsDictionary(self):
        params_dic = {
            "gamma": self.gamma,
            "pi": self.pi,
            "mu": self.mu,
            "Wx": self.Wx,
            "C": self.C,
            "Psi": self.Psi}

        return params_dic
def turnArr(arr):
	return np.array(arr).flatten().tolist()

def plotData(data,label,color=["r","g","b"]):
	M=len(data[0])
	pcaData=[]
	fig=plt.figure()
	pos=[221,222,223]
	

	for m in range(M):
		colorList=[color[i] for i in label]

		comData_m=[]
		ax=fig.add_subplot(pos[m],projection='3d')
		for n in range(len(data)):
			comData_m.append(np.array(np.r_[data[n][m][0],data[n][m][1]]).flatten())
		pca=PCA(n_components=3)
		pca.fit(comData_m)
		pcaData_m=pca.fit_transform(comData_m)
		xList=[pd[0] for pd in pcaData_m]
		yList=[pd[1] for pd in pcaData_m]
		zList=[pd[2] for pd in pcaData_m]
		
		print type(ax)
		ax.scatter(xList,yList,zList,c=colorList)

		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')
		
	plt.show()
		



def main():
    #generator = ArtificialDataGenerator([0.33,       0.33,       0.34],
    #                                   ["params1", "params2", "params3"])
    generator = ArtificialDataGenerator([0.15,       0.25,       0.6],
                                        [["params1", "params2", "params3"],["params2","params1","params3"],["params3","params2","params1"]])
    
    data = generator.generate(300)
    EM = PCCAEMalgorithmn(data, 3)

    #result=EM.calcSomeStep(100)
    result=EM.calcUntilNoChangeClustering()
    
    correct_cluster_labels = generator.get_z()



    

	
    return result , generator.z , data

'''
    #kmeans_results = generator.clusteringDataWithKmeans(3)

    # 過去の方法でやるやつ そんなにいらない
    from CausalPatternExtractor import CausalPatternExtractor
    from random import randint
    data = np.array(generator.get_data())
    print data.shape
    y1 = data[:, 0:2]
    y2 = data[:, 2:5]
    x = data[:, 5:9]
    frames_clusterLabels = [randint(0, 2) for i, _ in enumerate(x)]
    clusterLabel_frames = []
    for label_i in range(max(frames_clusterLabels) + 1):
        clusterLabel_frames.append([i for i, j in enumerate(frames_clusterLabels) if j == label_i])
    extractor = CausalPatternExtractor(y1, y2, x, clusterLabel_frames)
    step_sumErrors, frames_clusterLabels = extractor.extract(1000)
    print frames_clusterLabels
    #########

    serial_results = EM.calcUntilNoChangeClustering()
#    serial_results = EM.calcUntilNoChangeGamma()

    correct_cluster_labels = generator.get_z()

    from Plotter import plot3dData, matchClusterLabels  # ファイル名とか考えなおす

    correct_cluster_labels, serial_results = matchClusterLabels(correct_cluster_labels, serial_results, 3)
#    kmeans_results, serial_results = matchClusterLabels(kmeans_results, serial_results, 3)

#    serial_errors = []
#    for time_t, results in enumerate(serial_results):
#        serial_errors.append(np.count_nonzero(results - correct_cluster_labels))
#    print serial_errors
#    f = open("/data1/keisuke.kawano/results/generate_model_results2.txt", "a")
#    f.write("%s\n"%serial_errors)
#    f.close()
#
#

#    from Plotter import plotErrorChange
#    plotErrorChange(correct_cluster_labels, serial_results)
    plot3dData(generator.get_pcaedData(3),
               [np.array(correct_cluster_labels),
                np.array(serial_results[-1]),
                np.array(kmeans_results)],
               ["Truth", "Proposed", "Previous"])
#               np.array(frames_clusterLabels))

    params_dic = EM.getParamsDictionary()
    print params_dic.items()
    print EM.cluster_eigen_vals
    print EM.cluster_eigen_vecs


if __name__ == "__main__":
    main()

	'''
