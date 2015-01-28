import numpy
from math import e

def sigmoid(s):
	s = float(s)
	return 1.0/(1+e**(-s))

	
def train_trainset(T,theta):
	train_set = numpy.genfromtxt('hw3_train.dat')
	y = train_set[:,-1]
	x = train_set[:,:-1]
	x_insert = numpy.insert(x,0,values = 1, axis = 1)
	x_insert = numpy.mat(x_insert)
	y = numpy.mat(y)
	#print y
	#print y.shape
	m,n = numpy.shape(x_insert)
	w = numpy.zeros((n,1))
	for k in range(T):
		result = numpy.zeros((n,1))
		for j in range(m):
			result += sigmoid(-y.item(j)*x_insert[j]*w)*(-y.item(j)*x_insert[j]).T
		result = result*1.0/m
		w = w - theta*result
	return w

def stochastic_gradient_descent(T,theta):
	train_set = numpy.genfromtxt('hw3_train.dat')
	y = train_set[:,-1]
	x = train_set[:,:-1]
	x_insert = numpy.insert(x,0,values = 1, axis = 1)
	x_insert = numpy.mat(x_insert)
	y = numpy.mat(y)
	#print y
	#print y.shape
	m,n = numpy.shape(x_insert)
	w = numpy.zeros((n,1))
	for k in range(T):
			j = k%m
			result = sigmoid(-y.item(j)*x_insert[j]*w)*(-y.item(j)*x_insert[j]).T
			w = w - theta*result
	return w
	
if __name__ == "__main__":
	#w = train_trainset(2000,0.01)
	w = stochastic_gradient_descent(2000,0.001)
	test_set = numpy.genfromtxt('hw3_test.dat')
	y = test_set[:,-1]
	x = test_set[:,:-1]
	x_insert = numpy.insert(x,0,values = 1, axis = 1)
	x_insert = numpy.mat(x_insert)
	m,n = numpy.shape(x_insert)
	y = numpy.mat(y)
	y_c = numpy.sign(x_insert * w)
	result = 0
	for i in range(m):
		if y_c.item(i) != y.item(i):
			result += 1.0
	result = result / m
	print result
			
		
	
	