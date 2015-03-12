import numpy as np
import math
import random
def loadDataSet(filename):
        fr = open(filename)
        dataSet = []
	for line in fr.readlines():
		curline = line.strip().split()
		fltline = map(float, curline)
		dataSet.append(fltline)
	fr.close()
	return dataSet

def randCent(dataSet, k):
	n = dataSet.shape[0]
	m = dataSet.shape[1]
	#centriods = np.mat(zeros((k,n)))
	#np.random.shuffle(dataSet)
	#centriods = dataSet[:k,:]
	centriods = np.array(np.zeros((k,m)))
	index = []
	for i in range(k):
		c = random.randint(0,n-1)
		while c in index:
			c = random.randint(0,n-1)
		index.append(c)
		centriods[i] = dataSet[c]
	#print centriods
	return centriods

def distEclud(vecA, vecB):
	return math.sqrt(sum(np.power(vecA-vecB, 2)))

def kMeans(dataSet, k, createCent = randCent):
	m = dataSet.shape[0]
	clusterAssment = np.mat(np.zeros((m,2)))
	centroids = createCent(dataSet, k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			minDist = float('inf'); minIndex = -1
			for j in range(k):
				distJI = distEclud(centroids[j,:], dataSet[i,:])
				if distJI < minDist:
					minDist = distJI; minIndex = j
			if clusterAssment[i,0] != minIndex: clusterChanged = True
			clusterAssment[i,:] = minIndex, minDist**2
		for cent in range(k):
			ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
			#print 'ptsInClust:'
			#print ptsInClust
			centroids[cent,:] = np.mean(ptsInClust, axis = 0)
	#print centroids
	sqdistance = 0.0
	for cent in range(k):
		ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
		pts_num = ptsInClust.shape[0]
		diffs = np.tile(centroids[cent,:], (pts_num,1)) - ptsInClust
		#print "diff"
		#print diffs
		sqdiffs = diffs ** 2
		sqdistance += sum(sqdiffs.sum(axis = 1))
	#print sqdistance/m
	return sqdistance/m
	
			
	
if __name__ == "__main__":
	dataset = loadDataSet('hw4_kmeans_train.dat')
	d_a = np.array(dataset)
	#centriods = randCent(d_a, 2)
	result = 0.0
	for i in range(500):
		result += kMeans(d_a, 10)
	print result/500
	