from numpy import *
from math import sqrt,log
x = []
y = []
x_test = []
y_test = []
fobj = open("hw2_adaboost_train.dat",'r')
fobj_test = open("hw2_adaboost_test.dat",'r')
for line in fobj:
	line_x = [float(s) for s in line.split()[:-1]]
	line_y = int(line.split()[-1])
	x.append(array(line_x))
	y.append(line_y)
fobj.close()

for line in fobj_test:
	line_x = [float(s) for s in line.split()[:-1]]
	line_y = int(line.split()[-1])
	x_test.append(array(line_x))
	y_test.append(line_y)
fobj_test.close()


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
	retArray = ones((shape(dataMatrix)[0],1))
	if threshIneq == -1:
		retArray[dataMatrix[:,dimen] >= threshVal] = -1
	else:
		retArray[dataMatrix[:,dimen] < threshVal] = -1
	return retArray

def buildStump(dataArr, classLabels, D, threashs):
	dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
	m,n = shape(dataMatrix)
	bestStump = {}; bestClasEst = mat(zeros((m,1)))
	minError = inf
	for i in range(n):
		#these can sent into out of this function
		threash = threashs[i]
		for threshVal in threash:
			for inequal in [-1,1]:
				predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
				errArr = mat(ones((m,1)))
				errArr[predictedVals == labelMat] = 0
				weightedError = D.T*errArr
				if weightedError < minError:
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump,minError,bestClasEst

def adaboostTrain(dataArr, classLabels, numIter):
	dataMatrix = mat(dataArr)
	m, n = shape(dataMatrix)
	weakClassArr = []
	aggClassEst = mat(zeros((m,1)))
	threashs = {}
	for i in range(n):
		threashs[i]  = [float("-inf")]
		d = sort(dataMatrix, axis = i)
		choice = d[:,i]
		p,q = shape(choice)
		for t in (range(p-2)):
			threashs[i].append((choice[t]+choice[t+1])/2)
	U = mat((ones((m,1)))/m)
	episilon_q = []
	for i in range(numIter):
		bestStump,error,classEst = buildStump(dataArr, classLabels, U, threashs)
		episilon = sum(multiply(U,(classEst != mat(classLabels).T)))/ sum(U)
		delta = sqrt((1.0 - episilon)/episilon)
		alpha = log(delta)
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		for t in range(m):
			if classEst[t] != classLabels[t]:
				U[t] = U[t] * delta
			else:
				U[t] = U[t] / delta
		aggClassEst += alpha * classEst
		aggErrors  = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))
		episilon_q.append(episilon)
	#episilon_q.sort()
	#print episilon_q[0]
	return weakClassArr
classifierArray = adaboostTrain(x, y, 300)
test_data = mat(x_test)
m = shape(test_data)[0]
aggClassEst = mat(zeros((m,1)))
for i in range(len(classifierArray)):
	classEst = stumpClassify(test_data, classifierArray[i]['dim'], classifierArray[i]['thresh'], classifierArray[i]['ineq'])
	aggClassEst += classifierArray[i]['alpha'] * classEst

Eout = multiply(sign(aggClassEst) != mat(y_test).T, ones((m,1)))
print sum(Eout)/m
		
		
		
		
		
		
		
		
		
		
		
		
		