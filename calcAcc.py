import numpy as np
import itertools

def calculateAccuracy(truth,result,truth_unique,result_unique):
		ruleList=[[truth_unique[i],result_unique[i]] for i in range(len(truth_unique))]
		iterNum=min(len(truth),len(result))
		zeroInTruth=truth[-iterNum:].count(0)
		accuracy=0
		for i in range(1,iterNum+1):
			if [truth[-i],result[-i]] in ruleList:
				accuracy+=1
		accuracyRate=float(accuracy)/(iterNum-zeroInTruth)
		return accuracyRate
		
	
def accuracyRate(truth,result):

		result=np.array(result).flatten()
		uniqueItem_truth=list(np.unique(truth))
		uniqueItem_truth.remove(0)
		uniqueItem_result=np.unique(result)
		len_truth=len(uniqueItem_truth)
		len_result=uniqueItem_result.shape[0]
		if len_truth>=len_result:
			list_truth=list(uniqueItem_truth)
			fullPermutations_truth=list(itertools.permutations(list_truth,len_result))
			maxAccuracy=0
			for i in fullPermutations_truth:
				accuracy=calculateAccuracy(truth,result,i,list(uniqueItem_result))
				if accuracy>maxAccuracy:
					maxAccuracy=accuracy

		if len_truth<len_result:
			list_truth=list(uniqueItem_truth)
			fullPermutations_resngult=list(itertools.permutations(list(uniqueItem_result),len_truth))
			maxAccuracy=0
			for j in fullPermutations_result:
				accuracy=calculateAccuracy(truth,result,list_truth,j)
				if accuracy>maxAccuracy:
					maxAccuracy=accuracy

		return maxAccuracy
