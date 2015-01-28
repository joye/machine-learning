from random import uniform,random

def sign(x):
	if x > 0:
		return 1
	else:
		return -1

def generate_dataset():
	result = []
	for i in range(20):
		x = uniform(-1,1)
		if random() > 0.2:
			y = sign(x)
		else:
			y = -sign(x)
		#print x,y
		result.append((x,y))
	return result

def calculate(dataset,s,theta):
	Ein = 0.0
	for sample in dataset:
		if s*sign(sample[0]-theta) != sample[1]:
			Ein += 1.0
	return Ein/len(dataset)

def decision_stump(dataset,s):
	ping = len(dataset)
	theta_result = 0
	for i in range(len(dataset)-2):
		theta = (dataset[i][0]+dataset[i+1][0])/2.0
		Ein = calculate(dataset,s,theta)
		if Ein < ping:
			ping = Ein
			theta_result = theta
	return (ping,theta_result)
		
def parse_multi_dimen():
	fp = open("decision_stump_train.txt",'r')
	train_data = []
	for line in fp.readlines():
		train_data.append((line.strip().split()))
	fp.close()
	return train_data

def gen_dimen_dataset(train_data,i):
	a = [(float(t[i]),int(t[-1])) for t in train_data]
	return a

def test(dataset):
	sorted_dataset = sorted(dataset,key=lambda set: set[0])
	(Ein_pos,theta_pos) = decision_stump(sorted_dataset,1)
	(Ein_neg,theta_neg) = decision_stump(sorted_dataset,-1)
	if Ein_pos < Ein_neg:
		Ein_result = Ein_pos
		theta_result = theta_pos
		s_result = 1
	else:
		Ein_result = Ein_neg
		theta_result = theta_neg
		s_result = -1	
	return (theta_result,s_result,Ein_result)
	
if __name__ == "__main__":
	EIN = 0.0
	EOUT = 0.0
	for i in range(5000):
		dataset = generate_dataset()
		sorted_dataset = sorted(dataset,key=lambda set: set[0])
		(Ein_pos,theta_pos) = decision_stump(sorted_dataset,1)
		(Ein_neg,theta_neg) = decision_stump(sorted_dataset,-1)
		if Ein_pos < Ein_neg:
			Ein_result = Ein_pos
			theta_result = theta_pos
			s_result = 1
		else:
			Ein_result = Ein_neg
			theta_result = theta_neg
			s_result = -1
		Eout_result = 0.5 + 0.3 * s_result* (abs(theta_result)-1)
		EIN += Ein_result
		EOUT += Eout_result
	print EIN/5000
	print EOUT/5000

	