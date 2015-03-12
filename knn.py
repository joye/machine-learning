import numpy as np
import operator
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

def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis = 1)
    distances = sqDistance**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDistIndices[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]
    
if __name__ == '__main__':
    #train = loadDataSet('/Users/yewei/machine_learning/taiwan/hw4_knn_train.dat')
	train = loadDataSet('hw4_knn_train.dat')
	train_a = np.array(train)
	#print train_a
	train_inx = train_a[:,:-1]
	train_iny = train_a[:,-1]
    #test = loadDataSet('/Users/yewei/machine_learning/taiwan/hw4_knn_test.dat')
	test = loadDataSet('hw4_knn_test.dat')
	test_a = np.array(test)
	test_inx = test_a[:,:-1]
	test_iny = test_a[:,-1]
	result = []
	for t in test_inx:
		d = classify(t, train_inx, train_iny, 1)
		result.append(d)
	result_a = np.array(result)
	eout = 1.0*sum(result_a != test_iny)/len(test)
	print eout
    