import numpy as np
import scipy
import scipy.sparse as sparse
from numpy.linalg import eigh
import itertools
import cmath 
import timeit
import math

from matplotlib import pylab as plt

from scipy.optimize import fsolve

from sympy.combinatorics import SymmetricGroup
from sympy.combinatorics import Permutation

from scipy.interpolate import interp1d

X = [[0.0, 1.0],[1.0, 0.0]]
Y = [[0.0, -1.0j],[1.0j, 0.0]]
Z = [[1.0, 0.0],[0.0, -1.0]]
I = [[1.0, 0.0],[0.0, 1.0]]

nontrivletters = ['X', 'Y', 'Z']
sitebasisletters = ['I','X', 'Y', 'Z']

nontriv = [X, Y, Z]
sitebasis = [I, X, Y, Z]

zero = np.array([1.0, 0.0])
one = np.array([0.0, 1.0])

Xs = sparse.csr_matrix(np.array([[0.0, 1.0],[1.0, 0.0]]))
Ys = sparse.csr_matrix(np.array([[0.0, -1.0j],[1.0j, 0.0]]))
Zs = sparse.csr_matrix(np.array([[1.0, 0.0],[0.0, -1.0]]))
Is = sparse.csr_matrix(np.array([[1.0, 0.0],[0.0, 1.0]]))

L=10
eigvecs = np.loadtxt('chaoticvecsnew'+str(L)).view(complex)
eigvals = np.real(np.loadtxt('chaoticvalsnew'+str(L)).view(complex))

#times = np.arange(0.0, 80.0, 5.0)
times0 = np.arange(0.0, 5.0, 0.1)
times1 = np.arange(0.0, 420.0, 20.0)
times1a = np.arange(0.0, 120.0, 10.0)
times2 = np.arange(0.0, 150.0, 10.0)
times3 = np.arange(0.0, 200.0, 10.0)
times4 = np.arange(0.0, 250.0, 10.0)
#late_times = np.arange(50.0, 75.0, 5.0)
#times2= [10.25, 10.75, 11.25, 11.75,  12.25, 12.75,  13.25, 13.75, 14.25, 14.75]
#alltimes = np.arange(10.0, 15.0, 0.25)
#print alltimes
#basic functions 

def complexify(X): 
	return np.add(X, 1j*np.zeros(X.shape))

def multikron(A):
	prod =1
	for i in range(len(A)):
		prod = np.kron(prod, A[i])
	return prod


#evolving with ED for initial states with support on a limited set of eigenvalues 

def exact_diag_evolve(psi, t, coeffs, L): 
	evolved_state = np.zeros(2**L)
	for i in range(2**L): 
		evolved_state = np.add(evolved_state, coeffs[i]*np.exp(-1j*eigvals[i]*t)*eigvecs[:,i])
	return evolved_state

#calculating density matrices and entanglement entropy 

def density_mat_half_line(psi, a, L):
	rho = complexify(np.zeros(((2**a), (2**a))))
	
	for na in range(2**a): 
			for nap in range(2**a): 
					x =0 
					for nac in range(2**(L-a)): 
							x = x + psi[int(bin(na)[2:].zfill(a)+bin(nac)[2:].zfill(L-a),2)]*np.conjugate(psi[int(bin(nap)[2:].zfill(a)+bin(nac)[2:].zfill(L-a),2)])
					#print str(na)+str(mb)+'_'+str(nap)+str(mbp)+' '+str(x)
					rho[na][nap]=x
	return rho	

def density_mat_half_line_left(psi, a, L):
	rho = complexify(np.zeros(((2**a), (2**a))))
	
	for na in range(2**a): 
			for nap in range(2**a): 
					x =0 
					for nac in range(2**(L-a)): 
							x = x + psi[int(bin(nac)[2:].zfill(L-a)+bin(na)[2:].zfill(a),2)]*np.conjugate(psi[int(bin(nac)[2:].zfill(L-a)+bin(nap)[2:].zfill(a),2)])
					#print str(na)+str(mb)+'_'+str(nap)+str(mbp)+' '+str(x)
					rho[na][nap]=x
	return rho	

def partial_trace(rho, a, b):

	rho_new = complexify(np.zeros((2**b, 2**b)))
	for na in range(2**b): 
		for nb in range(2**b):
				x =0 
				for nc in range(2**(a-b)): 
						x= x+rho[int(bin(na)[2:].zfill(b)+bin(nc)[2:].zfill(a-b),2)][int(bin(nb)[2:].zfill(b)+bin(nc)[2:].zfill(a-b),2)]
				#print str(na)+str(mb)+'_'+str(nap)+str(mbp)+' '+str(x)
				rho_new[na][nb]=x
	return rho_new

def partial_trace_left(rho, a, b):

	rho_new = complexify(np.zeros((2**b, 2**b)))
	for na in range(2**b): 
		for nb in range(2**b):
				x =0 
				for nc in range(2**(a-b)): 
						x= x+rho[int(bin(nc)[2:].zfill(a-b)+bin(na)[2:].zfill(b),2)][int(bin(nc)[2:].zfill(a-b)+bin(nb)[2:].zfill(b),2)]
				#print str(na)+str(mb)+'_'+str(nap)+str(mbp)+' '+str(x)
				rho_new[na][nb]=x
	return rho_new

def entropy(densitymat): 
	p = np.linalg.eigvals(densitymat)
	ent = np.zeros(6)
	#one = 0.0
	for i in range(len(p)):
		#one = one+p[i]
		#print p[i] 
		if p[i]!=0:
			for j in range(6): 
				if j==0: 
					ent[j] = ent[j]- p[i]*np.log(p[i])
				else: 
					ent[j] = ent[j] + p[i]**(j+1)
	for j in range(1, 6): 
		ent[j] = -1.0/(j)*np.log(ent[j])

	#print one 
	return ent

#diagonal approximation 
def diag_approx(omega, na, n): 	
	ent = 0 

	omega_a = partial_trace(omega, L, na)
	omega_b = partial_trace_left(omega, L, L-na)

	num = np.trace(np.matmul(omega_a, omega_a))+ np.trace(np.matmul(omega_b, omega_b))

	return  -1.0/(n-1)*np.log(num)


#create initial product state, all Y up 

psi_list = []
for i in range(L): 
	psi_list.append([-1j/np.sqrt(2), 1/(np.sqrt(2))])
psi=multikron(psi_list) 


#find coefficients of initial state along all energy eigenstates 

coeffs = []
for i in range(2**L): 
	coeff= np.dot(np.conjugate(eigvecs[:,i]), psi)
	coeffs.append(coeff)

omega = np.zeros((2**L, 2**L))

for i in range(2**L):
	omega = omega + (np.abs(coeffs[i])**2)*np.outer(eigvecs[:, i], np.conjugate(eigvecs[:, i])) 



def phi(t): 
	spec =0 
	for i in range(2**L): 
		spec = spec + np.exp(1j*eigvals[i]*t)
	phi = spec/(2**L)

	return phi

cx = np.random.uniform()
cy = np.random.uniform()
cz = np.random.uniform()

#cx = 0.38601688099
#cy = 0.627067412372
#cz = 0.682561386402
O = cx*np.array(X) + cy*np.array(Y) +  cz*np.array(Z)
print cx
print cy
print cz
ops = [multikron([O, I, I, I, I, I, I, I, I, I]), multikron([O, O, O, I, I, I, I, I, I, I]), multikron([O, O, O, O, O, O, O, I, I, I])]
sizes = [1, 3, 7]

#ops = [ multikron([O, I, I, I, I , I , I, I, I, I, I, I]), multikron([O, O, O, I, I , I , I, I, I, I, I, I]),  multikron([O, O, O, O, O , I , I, I, I, I, I, I]), 
# multikron([O, O, O, O, O , O , O, I, I, I, I, I]), multikron([O, O, O, O, O , O , O, O, O, I, I, I]), multikron([O, O, O, O, O , O , O, O, O, O, O, O])]
#sizes = [1, 3, 5, 7, 9, 12]

#ops = [ multikron([O, I, I, I, I , I , I, I, I, I, I, I]),   multikron([O, O, O, I, I , I , I, I, I, I, I, I]), 
# multikron([O, O, O, O, O , O , O, O, O, I, I, I])]
#sizes = [1, 3, 9]

ops_initial = [np.dot(np.conjugate(psi), np.dot(ops[i], psi)) for i in range(len(ops))] 

ops_diag = [ np.trace(np.matmul(omega, ops[i])) for i in range(len(ops))]

print ops_initial
print ops_diag

with open('all_ops_initial_'+str(L)+'_Ystate_'+'_cx_'+str(cx)+'_cy_'+str(cy)+'_cz_'+str(cz)+'.npy', 'wb') as g:
	np.save(g, ops_initial)

with open('all_ops_diag_approx_'+str(L)+'_Ystate_'+'_cx_'+str(cx)+'_cy_'+str(cy)+'_cz_'+str(cz)+'.npy', 'wb') as g:
	np.save(g, ops_diag)

def half_line_time_ev(times, label): 


	allopstimes =[]
	allopstimes_approx = []
	phis = []
	for t in range(len(times)):  


		phi2 = np.abs(phi(times[t]))**2

		phis.append(phi2)

		psit= exact_diag_evolve(psi, times[t], coeffs, L)
		
		ops_t = []
		ops_t_approx = []

		for j in range(len(ops)): 
			ops_t.append(np.dot(np.conjugate(psit),np.dot(ops[j], psit)))

			app = ops_diag[j] + phi2*(ops_initial[j]-ops_diag[j]) 
			ops_t_approx.append(app)
	
		allopstimes.append(ops_t)
		allopstimes_approx.append(ops_t_approx)
		
	with open('allopstimes_'+str(L)+'_Ystate_maxtime_'+label+'_cx_'+str(cx)+'_cy_'+str(cy)+'_cz_'+str(cz)+'.npy', 'wb') as f:
		np.save(f,allopstimes)

	#with open('allopstimes_'+str(L)+'_E_'+str(E)+'_delta_'+str(delta)+'_equal_sym_maxtime_'+label+'_cx_'+cx+'_cy_'+cy+'_cz_'+cz+'.npy', 'rb') as f:
	#	allopstimes = np.load(f)

	plt.plot(times, phis, 'o-')
	plt.xlabel(r'$t$')
	plt.ylabel(r'$|\phi(t)|^2$')
	plt.show()

	averages = []


	for j in range(len(ops)):
		op_j = [ allopstimes[t][j] for t in range(len(times))]
		averages.append(np.mean(op_j))
		op_j_approx = [ allopstimes_approx[t][j] for t in range(len(times))]
		plt.plot(times, op_j, 'o-', label=str(sizes[j])+' e')
		plt.plot(times, op_j_approx, '>-', label=str(sizes[j])+' a')
		plt.xlabel(r'$t$')
		plt.ylabel(r'<O>')
		plt.legend(loc='best')
		plt.title('L='+str(L)+', Ystate')
		plt.show()


	print averages 

	

	
half_line_time_ev(times0, str(times1[-1]))

