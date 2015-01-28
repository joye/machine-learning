from random import uniform,random
from numpy import mat,linalg,genfromtxt,insert,shape,identity,sign


def generate_dataset():
	result = []
	for i in range(1000):
		x1 = uniform(-1,1)
		x2 = uniform(-1,1)
		y = x1**2 + x2**2 - 0.6
		if random() > 0.1:
			d = sign(y)
		else:
			d = -sign(y)
		result.append((1,x1,x2,d))
	return result

def Linear_Regress(Mat_data):
	#Mat_data = mat(dataset)
	xMat = Mat_data[:,:-1]
	yMat = Mat_data[:,-1]
	xTx = xMat.T * xMat
	if linalg.det(xTx) == 0.0:
		print "This matrix is singuar, cannot do inverse"
		return
	ws = xTx.I * xMat.T * yMat
	return ws

def transform(dataset):
	transformed = []
	for data in dataset:
		x1 = data[1]
		x2 = data[2]
		y  = data[3]
		transformed.append((1,x1,x2,x1*x2,x1**2,x2**2,y))
	return transformed

def regularized_linear_regress(x,y,lamb):
	xTx = x.T * x
	m,n = shape(xTx)
	if m != n:
		print "error"
		return
	Iden = lamb * identity(m)
	xTx = xTx + Iden
	ws = xTx.I * x.T * y.T
	return ws
	

if __name__ == "__main__":
	Ein = 0
	Eout = 0
	train_set = genfromtxt('hw4_train.dat')
	y = train_set[:,-1]
	x = train_set[:,:-1]
	x_insert = insert(x,0,values = 1, axis = 1)
	x_insert = mat(x_insert)
	m,n = shape(x_insert)
	test_set = genfromtxt('hw4_test.dat')
	y_t = test_set[:,-1]
	x_t = test_set[:,:-1]
	x_t = insert(x_t,0,values = 1, axis = 1)
	x_t = mat(x_t)
	y_t = mat(y_t)
	y = mat(y)
	lamb = [2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
	Ein_min = 1
	Eout_min = 1
	lamb_min = 10
	for ttt in lamb:
		ttt = 10**ttt
		Ein = 0
		Eout = 0
		w = regularized_linear_regress(x_insert,y,ttt)
		y_c = sign(x_insert*w)
		for i in range(m):
			if y_c.item(i) != y.item(i):
				Ein += 1.0	
		y_t_c = sign(x_t * w)
		m_t,n_t = shape(x_t)
		for i in range(m_t):
			if y_t_c.item(i) != y_t.item(i):
				Eout += 1.0
		Ein = Ein/m
		Eout = Eout/m_t
		if Eout < Eout_min:
			Eout_min = Eout
			lamb_min = ttt
		if Eout == Eout_min:
			if ttt > lamb_min:
				lamb_min = ttt
	print Eout_min, lamb_min
		
	