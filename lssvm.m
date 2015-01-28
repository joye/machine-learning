clc;
clear all;
m = load('hw2_lssvm_all.dat');
train = m(1:400,:);
test = m(401:500,:);
X_train = train(:,1:10);
Y_train = train(:,11);

X_test = test(:,1:10);
Y_test = test(:,11);


[kk, n] = size(X_train);
gamma_a = [32, 2, 0.125];
lamb_a  = [0.001,1,1000];

for gamma_index = 1:3;
	for lamb_index = 1:3;
		gamma = gamma_a(gamma_index)
		lamb  = lamb_a(lamb_index) 
		K = zeros(kk,kk);
		for i = 1:kk;
			for j = 1:kk;
				K(i,j) = exp(-gamma * ((norm(X_train(i,:) - X_train(j,:)))^2));
			endfor
		endfor
		beta = inverse(lamb * eye(kk) + K) * Y_train;

		Ein = 0;
		for t = 1:kk;
			l = 0;
			for j = 1:kk;
				l += beta(j) * exp(-gamma * ((norm(X_train(t,:) - X_train(j,:)))^2));
			endfor
			Ein += (Y_train(t) != sign(l));
		endfor
        Ein/kk
		
		[jj, ll] = size(X_test);
		Eout = 0;
		for t = 1:jj;
			l = 0;
			for j = 1:kk;
				l += beta(j) * exp(-gamma * ((norm(X_test(t,:) - X_train(j,:)))^2));
			endfor
			Eout += (Y_test(t) != sign(l));
		endfor
		Eout/jj
	endfor
endfor



