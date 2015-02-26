from numpy import *

#global branch_func
def loadDataSet(filename):
	fr = open(filename)
	dataSet = []
	for line in fr.readlines():
		curline = line.strip().split()
		fltline = map(float, curline[:-1])
		fltline.append(int(curline[-1]))
		dataSet.append(fltline)
	fr.close()
	return dataSet

def createTree(dataset, depth):
	classList = [example[-1] for example in dataset]
	dataMat = mat(dataset)
	m,n = shape(dataMat)
	retTree = {}
	if classList.count(classList[0]) == len(classList):
		#retTree['is_leaf'] = True;
		return classList[0]
	feature_set = []
	for i in range(n-1):
		featList = [example[i] for example in dataset]
		feature_set.append(len(set(featList)))
	if sum(feature_set) == (n-1):
		#retTree['is_leaf'] = True;
		return classList[0]
	#retTree['is_leaf'] = False;
	bestSplit = chooseBestSplit(dataMat)
	#branch_func += 1
	#print "begin to branch"
	retTree['spInd'] = bestSplit['dim']
	retTree['spVal'] = bestSplit['thresh']
	retTree['left'] = createTree(bestSplit['left'], depth+1)
	retTree['right']= createTree(bestSplit['right'], depth+1)
	return retTree

def Calc_GiniIndex(mat):
	m, n = shape(mat)
	if m != 0:
		u = 1.0*sum(mat[:,-1] == 1) / m
		gini = 2.0*u*(1-u)
		return gini * m
	else:
		return 0
		
def chooseBestSplit(dataMat):
	#numFeatures = len(dataSet[0]) - 1
	m,n = shape(dataMat)
	threashs = {}
	bestsplit = {}
	b_x = inf
	for i in range(n-1):
		threashs[i]  = [float("-inf")]
		d = sort(dataMat, axis = i)
		choice = d[:,i]
		p,q = shape(choice)
		for t in range(p-1):
			threashs[i].append((choice[t]+choice[t+1])/2)
		threashs[i].append(float("inf"))
	for i in range(n-1):
		threash = threashs[i]
		for threshVal in threash:
			mat0 = dataMat[nonzero(dataMat[:,i] >= threshVal)[0],:][0]
			mat1 = dataMat[nonzero(dataMat[:,i] <  threshVal)[0],:][0]
			gini_result = Calc_GiniIndex(mat0) + Calc_GiniIndex(mat1)
			if gini_result < b_x:
				bestsplit['dim'] = i
				bestsplit['thresh'] = threshVal
				bestsplit['left'] = mat1.copy().tolist()
				bestsplit['right'] = mat0.copy().tolist()
				b_x = gini_result
	return bestsplit	

def predict(data, tree):
	#for data in dataset:
	if isinstance(tree, float):
		return tree
	else:
		spindex = tree['spInd']
		spVal = tree['spVal']
		if data[spindex] >= spVal:
			return predict(data, tree['right'])
		else:
			return predict(data, tree['left'])
		
if __name__ == '__main__':
	dataSet = loadDataSet('hw3_train.dat')
	testSet = loadDataSet('hw3_test.dat')
	y = [example[-1] for example in dataSet]
	y_t = [example[-1] for example in testSet]
	branch_func = 0
	retTree = createTree(dataSet,0)	
	print retTree
	e = []
	for data in dataSet:
		e.append(predict(data,retTree))
	#print e
	print 1.0 * sum(y != e) / len(y)
	e_t = []
	for data in testSet:
		e_t.append(predict(data,retTree))
	print e_t
	eout = 0
	for ind in range(len(e_t)):
		if e_t[ind] != y_t[ind]:
			eout += 1
	print 1.0 * eout / len(y_t)
	#print branch_func
	#calculate_branch(retTree)
	#chooseBestSplit(dataSet)
	

	

