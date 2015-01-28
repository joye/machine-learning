import numpy as np
from math import e
def hessian(u,v):
	delta_u2 = e**u + v*v*(e**(u*v))+2 #e**u + v2*e**uv + 2
	delta_v2 = 4*(e**(2*v))+u*u*(e**(u*v))+4 #4*e**2v+u2*e**uv+4
	delta_uv = e**(u*v)+u*v*(e**(u*v))-2
	return np.matrix([[delta_u2,delta_uv],[delta_uv,delta_v2]])

def jacobian(u,v):
	delta_u = e**u+v*(e**(u*v))+2*u-2*v-3
	delta_v = 2*(e**(2*v))+u*(e**(u*v))-2*u+4*v-2
	return np.array([delta_u,delta_v])

def newton(u,v):
	hessian_matrix = hessian(u,v)
	tic = np.matrix([u,v])
	tic = tic - hessian_matrix.I.dot(jacobian(u,v))
	u = tic[0,0]
	v = tic[0,1]
	return u,v

def calc_E(u,v):
	E = e**u+e**(2*v)+e**(u*v)+u**2-2*u*v+2*(v**2)-3*u-2*v
	return E

if __name__ == "__main__":
	u = 0
	v = 0
	for i in range(5):
		u_new,v_new = newton(u,v)
		u,v = u_new,v_new
	result = calc_E(u,v)
	print result