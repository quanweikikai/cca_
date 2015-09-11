#!-*-coding:utf-8-*-

from ArtificialDataGenerator import ArtificialDataGenerator
import numpy as np
import random
import warnings
import sys
import multiprocessing as mp
import functools

def calcMultivariateNormal(x, mean, covariance ):
	# the x and mean must have the shape (N,1)!
	covariance_inv=np.linalg.inv(covariance)
	n=mean.shape[0]
	det = (2*np.pi)**n*np.linalg.det(covariance) 
	return (1/float(det))**0.5 * np.exp(-0.5* (x-mean).T * covariance_inv * (x-mean))

def calcFunc(Y,Ytm,Xtm,Wx,Wy):
	# Y------> (N,1)
	# Ytm----> (My,1)
	# Xtm----> (Mx,1)
	# Wx-----> (N,Mx)
	# Wy-----> (N,My)
	lenY=len(Y)
	lenYtm=len(Ytm)
	lenXtm=len(Xtm)

	Y=np.matrix(Y).reshape(lenY,1)
	Ytm=np.matrix(Ytm).reshape(lenYtm,1)
	Xtm=np.matrix(Xtm).reshape(lenXtm,1)
	Wx=np.matrix(Wx)
	Wy=np.matrix(Wy)
	return Y-Wx*Xtm-Wy*Ytm

def calcG(Y,Ytm,Wy,mu):
	# Y------> (N,1)
	# Ytm----> (My,1)
	# mu-----> int 
	# Wy-----> (N,My)
	lenY=len(Y)
	lenYtm=len(Ytm)

	Y=np.matrix(Y).reshape(lenY,1)
	Ytm=np.matrix(Ytm).reshape(lenYtm,1)
	Wy=np.matrix(Wy)
	return Y-Wy*Ytm

def calcH(Y,Xtm,Wx,mu):
	# Y------> (N,1)
	# Xtm----> (Mx,1)
	# Wx-----> (N,Mx)
	# mu-----> int 
	lenY=len(Y)
	lenXtm=len(Xtm)

	Y=np.matrix(Y).reshape(lenY,1)
	Xtm=np.matrix(Xtm).reshape(lenXtm,1)
	Wx=np.matrix(Wx)
	return Y-Wx*Xtm


def calcGamma(Y, Ytm, Xtm, K,  Wx, Wy, mu, cov, PI):
	# x------> (N,1)
	# Ytm----> (My,1)
	# Xtm----> (Mx,1)
	# K------> N
	# Wx[]---> (N,Mx)  k items list 
	# Wy[]---> (N,My)  k items list
	# mu[]---> (N,1)   k items list
	# PI[k]--> k item list
	# cov[k]-> (N,N)   k item list

    weighted_pdf = []
    for k in range(K):
        x = calcFunc(Y,Ytm,Xtm,Wx[k],Wy[k]) 
        weighted_pdf.append((PI[k] * calcMultivariateNormal(x, mu[k], cov[k]))[0,0])
    return [float(weight)/sum(weighted_pdf) for weight in weighted_pdf]


class PCCAEMalgorithmn(object):
	# data is like :
	# [[Y , Ytm, Xtm],
	#  [            ],
	#  [...           ]
    def __init__(self, data, K):
        self.data = [[np.matrix(item[0]),np.matrix(item[1]),np.matrix(item[2])] for item in data]
        self.K = K
        dimension_1 = data[0][0].shape[0]
        dimension_2 = data[0][1].shape[0]
        dimension_3 = data[0][2].shape[0]
		
	self.PI=[1.0/K for i in range(K)]
	self.Wy=[np.matrix(np.random.rand(dimension_1,dimension_2)) for j in range(K)]
	self.Wx=[np.matrix(np.random.rand(dimension_1,dimension_3)) for l in range(K)]

	self.dataNum=len(self.data)
	self.mean=[]
	self.cov=[]
	self.likeh=-100000000000000000
	self.covSum=100000000000000000
	self.Num=0
	for i in range(K):
		data_x=[(calcFunc(d[0],d[1],d[2],self.Wx[i],self.Wy[i])).T for d in data ]
		choice=np.random.choice(self.dataNum,self.dataNum/2,replace=False)
		randomData=[data_x[i] for i in choice]
		#concatenate all the data items
		choiced_x=np.concatenate(randomData)
		self.mean.append(np.matrix(np.mean(choiced_x,axis=0).T))
		self.cov.append(np.matrix(np.cov(choiced_x.T)))

    def calcOneStep(self):
	sys.stdout.write("\r Estep : calc gamma")
	sys.stdout.flush()
	# calc the w
	w=[]
	for dataItem in self.data:
		w.append(calcGamma(dataItem[0],dataItem[1],dataItem[2],self.K,self.Wx,self.Wy,self.mean,self.cov,self.PI))
		
        sys.stdout.write("\r Mstep : calc PI      ")
        sys.stdout.flush()
	PI_new=np.mean(np.array(w),axis=0).tolist()

        #record all Ni
	NList=[i*self.dataNum for i in self.PI]

        sys.stdout.write("\r Mstep : calc mu      ")
        sys.stdout.flush()

	muList=[]
	omiga=np.array(w)
	############################################################
	#from here add the following lines to modify the omiga to binary ones
	oShape=omiga.shape
	iniOmiga=np.zeros(oShape)
	maxIndex=np.max(omiga,axis=1)
	for i in range(oShape[0]):
		iniOmiga[i,maxIndex[i]]=1

	self.omiga=iniOmiga

    ############################################################
	self.omiga=omiga
	d_1=self.data[0][0].shape[0]
	d_2=self.data[0][1].shape[0]
	d_3=self.data[0][2].shape[0]

	for j in range(self.K):
		tmp_mu=np.matrix(np.zeros((d_1,1)))
		for i in range(self.dataNum):
			tmp_mu+=omiga[i,j]*calcFunc(self.data[i][0],self.data[i][1],self.data[i][2],self.Wx[j],self.Wy[j])
		muList.append(1.0/NList[j]*tmp_mu)

        sys.stdout.write("\r Mstep : calc Wx      ")
        sys.stdout.flush()

	Wx_new=[]
	for j in range(self.K):
		tmp_Wx_1=np.matrix(np.zeros((d_1,d_3)))
		tmp_Wx_2=np.matrix(np.zeros((d_3,d_3)))
		for i in range(self.dataNum):
			tmp_Wx_1+=omiga[i,j]*calcG(self.data[i][0],self.data[i][1],self.Wy[j],self.mean[j])*(self.data[i][2].T)
			tmp_Wx_2+=omiga[i,j]*self.data[i][2]*(self.data[i][2].T)

		Wx_new.append(tmp_Wx_1*(np.linalg.inv(tmp_Wx_2)))
			
        sys.stdout.write("\r Mstep : calc Wy      ")
        sys.stdout.flush()

	Wy_new=[]
	for j in range(self.K):
		tmp_Wy_1=np.matrix(np.zeros((d_1,d_2)))
		tmp_Wy_2=np.matrix(np.zeros((d_2,d_2)))
		for i in range(self.dataNum):
			tmp_Wy_1+=omiga[i,j]*calcH(self.data[i][0],self.data[i][2],self.Wx[j],self.mean[j])*(self.data[i][1].T)
			tmp_Wy_2+=omiga[i,j]*self.data[i][1]*(self.data[i][1].T)

		Wy_new.append(tmp_Wy_1*(np.linalg.inv(tmp_Wy_2)))
		
        sys.stdout.write("\r Mstep : calc cov      ")
        sys.stdout.flush()

	cov_new=[]
	for j in range(self.K):
		tmp_cov=np.matrix(np.zeros((d_1,d_1)))
		for i in range(self.dataNum):
			tmp_func=calcFunc(self.data[i][0],self.data[i][1],self.data[i][2],self.Wx[j],self.Wy[j])-self.mean[j]
			tmp_cov+=omiga[i,j]*tmp_func*(tmp_func.T)
		cov_new.append(1.0/NList[j]*tmp_cov)

        sys.stdout.write("\r Mstep End      ")
        sys.stdout.flush()
	self.mean=muList
	self.cov=cov_new
	self.Wx=Wx_new
	self.Wy=Wy_new
	############################
	#modified here!!
	#self.PI=PI_new
	self.PI_new=PI_new
	self.PI=[1.0/self.K for i in range(self.K)]
	###########################
	self.Num+=1

		
	return self.likehood()

    def likehood(self):
	likeh=0.0
	for i in range(len(self.data)):
		tmp=0.0
		for j in range(self.K):
			x=calcFunc(self.data[i][0],self.data[i][1],self.data[i][2],self.Wx[j],self.Wy[j])
			tmp+=(self.omiga[i,j]*calcMultivariateNormal(x, self.mean[j], self.cov[j] ))[0,0]
		likeh+=np.log(tmp)
	return likeh

    def run(self,threshold=0):
	likeh=self.calcOneStep()
	covSum=self.covSum
	self.calcCovSum()
	#if covSum-self.covSum>threshold:
	if likeh-self.likeh>threshold or self.Num<100:
		self.likeh=likeh
		print likeh  , "******************"#, self.cov
		print "covariance sum:",self.covSum,"*******************",self.PI_new
		self.run()
	else: print likeh-self.likeh

    def calcCovSum(self):
		index=np.argmax(self.omiga,axis=1)
		covSum=0
		for i in range(len(self.data)):
			j=index[i]
			tmp=np.sum(np.array(calcFunc(self.data[i][0],self.data[i][1],self.data[i][2],self.Wx[j],self.Wy[j]))**2)
			covSum+=tmp
		self.covSum=covSum

		
#if __name__=="__main__":
'''
data=[[np.random.rand(10,1),np.random.rand(10,1),np.random.rand(10,1)] for i in range (1000)]
a=PCCAEMalgorithmn(data,3)
a.run()
'''
		



			

		

