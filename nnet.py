import random
import numpy as np
from math import exp
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
	
def tanh_f(s):
	a1 = np.array(map(exp, s)) 
	a2 = np.array(map(exp, -s))
	a = a1 - a2
	b = a1 + a2
	#b = map(exp, s) + map(exp, -s)
	return a/b
	
def nnet_train(train_set, T, eta, M, r):
	d = len(train_set[0]) - 1
	w = {0: np.random.uniform(-r, r,(d+1,M)), 1: np.random.uniform(-r,r,(M+1,1))}
	#w_2 = np.random.rand(M+1,1)
	N = len(train_set)
	for t in range(0,T+1):
		x_train = {}
		s = {}
		delta = {}
		n = random.randint(0, N-1)
		x = train_set[n][:-1]
		y = train_set[n][-1]
		layer = 2
		x.insert(0,1)
		x_train[0] = np.array(x)
		for i in range(layer):  #forward
			x_m = np.mat(x_train[i])
			w_m = np.mat(w[i])
			s[i] = x_m * w_m
			x_train[i+1] = np.insert(np.array(np.tanh(s[i]))[0],0,1)
		delta[layer] = -2 * (y - x_train[layer][1]) * (1 - np.power(x_train[layer][1],2)) #backward
		delta[layer-1] = delta[layer] * np.multiply(w[1][1:].T, (1 - np.power(x_train[1][1:],2)))
		#backward completion
		#print eta
		#print w[0]
		w[0] = w[0]-eta * (np.mat(x_train[0]).T) * delta[layer - 1]
		w[1] = w[1]-eta * (np.mat(x_train[1]).T) * delta[layer]
	return w

def predict(w, x):
	s1 = np.tanh(x * w[0])
	s1_insert = np.insert(s1, 0, 1, axis = 1)
	s2 = np.tanh(s1_insert * w[1])
	return np.sign(s2)
	
if __name__ == '__main__':
	#train_set = loadDataSet('/Users/yewei/Downloads/taiwan/hw4_nnet_train.dat')
	train_set = loadDataSet('hw4_nnet_train.dat')
	test_set = loadDataSet('hw4_nnet_test.dat')
	num = len(test_set)
	test_m = np.mat(test_set)
	x_test = test_m[:,:-1]
	y_test = test_m[:,-1]
	x_insert_t = np.insert(x_test, 0, 1, axis = 1)
	#print train_set
	min_eout = float('inf')
	min_M = 1
	for M in [1,6,11,16,21]:
		e_out = 0.0
		for i in range(500):
			w = nnet_train(train_set, 50000, 0.1, M, 0.1)
			#print w
			result = predict(w, x_insert_t)
			e_out += int(sum(result != y_test))*1.0/num
		if(e_out < min_eout):
			min_eout = e_out
			min_M = M
	print min_M
	#print int(sum(result != y_test))*1.0/num
	