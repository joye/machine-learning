import numpy
import scipy
from random import shuffle
train_data = numpy.genfromtxt('pla_train.dat')
#numpy.random.shuffle(train_data)


pocket_train_data = numpy.genfromtxt('pocket_train.dat.txt')
pocket_test_data  = numpy.genfromtxt('pocket_test.dat.txt')
def sign(w,x):
	temp = x.dot(w)
	result = []
	for i in temp:
		if i > 0:
			result.append(1.0)
		else:
			result.append(-1.0)
	return result

def mistake_indx(y_temp,y):
	#index = -1
	result = []
	for i in range(len(y_temp)):
		if(y_temp[i]!=y[i]):
			result.append(i)
			#break
	return result

def mistake_cal(y_temp,y):
	result = 0
	for i in range(len(y_temp)):
		if y_temp[i] != y[i]:
			result += 1
	return result
	
def pocket(w,x,y,theta):
	w_result = w
	sign_result = sign(w,x)
	ping = mistake_cal(sign_result,y)
	#print ping
	index = mistake_indx(sign_result,y)
	for j in range(100):
		shuffle(index)
		#print index
		w = w+x[index[0]].dot(y[index[0]])
		sign_result = sign(w,x)
		pong = mistake_cal(sign_result,y)	
		index = mistake_indx(sign_result,y)
		#print ping,pong
		if ping <= pong:
			continue
		else:
			w_result = w
			ping = pong
		#w_result = w
	print w_result
	return w_result
		
def pla(w,x,y,theta):
	sign_result = sign(w,x)
	index = mistake_indx(sign_result,y)
	iter_num = 1
	while len(index) != 0:	
		for i in range(x.shape[0]):
			if i in index:
				iter_num = iter_num + 1
				w = w+theta*(x[i].dot(y[i]))
				sign_result = sign(w,x)
				index = mistake_indx(sign_result,y)
	return iter_num

def pla_test(train_data):		
	average_iter = 0
	for i in range(2000):
		numpy.random.shuffle(train_data)
		y = train_data[:,-1]
		x = train_data[:,:-1]
		x_insert = numpy.insert(x,0,values = 1, axis = 1)
		w = numpy.array([0,0,0,0,0]) # initialize w 
		iter_num = pla(w,x_insert,y,1)
		average_iter += iter_num
	print average_iter/2000
	#average_iter = pla(w,x_insert,y,1)
	#print average_iter
	return average_iter
		
def pocket_test(train_data,test_data):
	result = 0
	for i in range(2000):
		print i
		numpy.random.shuffle(train_data)
		y = train_data[:,-1]
		x = train_data[:,:-1]
		x_insert = numpy.insert(x,0,values = 1, axis = 1)
		w = numpy.array([0,0,0,0,0]) # initialize w 
		result_w = pocket(w,x_insert,y,1)
		y_test = test_data[:,-1]
		x_test = test_data[:,:-1]
		x_test_insert = numpy.insert(x,0,values = 1, axis = 1)
		# print w.shape
		# print x_test_insert.shape
		test_result = sign(w,x_test_insert)
		result += mistake_cal(test_result,y_test)*1.0/y_test.shape[0]
	result = result / 2000
	return result

if __name__ == "__main__":
	error_rate = pocket_test(pocket_train_data,pocket_test_data)
	print error_rate
	#iter = pla_test(train_data)
	
	
	