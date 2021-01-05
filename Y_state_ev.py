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
times5 = np.arange(250.0, 350.0, 10.0)
times6 = np.arange(350.0, 450.0, 10.0)
plottimes5 = np.arange(0.0, 350.0, 10.0)
plottimes6 = np.arange(0.0, 450.0, 10.0)
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



def half_line_time_ev(times, na): 

	
	#omega_small = np.array(omega)*1.0/norm


	diagval = np.exp(-diag_approx(omega, na, 2))

	allentstimes =[]
	allentstimes_approx = [] 

	for t in range(len(times)):  
		phit= phi(times[t])

		psit= exact_diag_evolve(psi, times[t], coeffs, L)
		

		rho = density_mat_half_line(psit, na, L)
		Sa = entropy(rho)		

		allentstimes.append(Sa[1])
		allentstimes_approx.append(-np.log(diagval + (np.abs(phit)**4)*(np.exp(-allentstimes[0])-diagval)))
		
		
	
	plt.plot(times, allentstimes, 'o-')
	plt.plot(times, allentstimes_approx, '>-')
	plt.xlabel('t')
	plt.ylabel(r'$S_n$')
	#plt.legend()
	plt.title('L='+str(L)+', A = '+str(na)+', Y state, time-evolved entropies')
	plt.show()


#half_line_time_ev(times0, 5)
#half_line_time_ev(times0, 4)
#half_line_time_ev(times0, 3)
#half_line_time_ev(times0, 2)


def plotvals(times, allentstimes):

	plotvals =[]
	for na in range(1, L):
		timesarray= []
		for t in range(len(times)):
			timesarray.append(allentstimes[t][na-1][1])
		plotvals.append(np.mean(timesarray,axis=0))

	return plotvals 


def time_average_append(times, label, entropies_sofar, plottimes): 

	allentstimes =entropies_sofar

	for t in range(len(times)):  
		psit= exact_diag_evolve(psi, times[t], coeffs, L)
		entropies_na = []
		for na in range(1, L): 
			if na < L/2+1:
				rho = density_mat_half_line(psit, na, L)
				Sa = entropy(rho)		
				entropies_na.append(Sa)
			else: 
				rho = density_mat_half_line_left(psit, na, L)
				Sa = entropy(rho)		
				entropies_na.append(Sa)
		print 'here'

		allentstimes = np.concatenate((allentstimes, [entropies_na]), axis=0)
		
	with open('allentstimes_'+str(L)+'_Ystate_maxtime_'+label+'.npy', 'wb') as f:
		np.save(f,allentstimes)
	print allentstimes.shape
	print len(allentstimes)
	#with open('allentstimes_'+str(L)+'_Ystate_maxtime_'+label+'.npy', 'rb') as f:
	#	allentstimes= np.load(f)
	
	#with open('allentstimes_'+str(L)+'_E_'+str(E)+'_delta_'+str(delta)+'_equal_sym.npy', 'rb') as f:
	#	allentstimes = np.load(f)

	renyi2 = plotvals(plottimes, allentstimes)

	plt.plot(range(1,L), renyi2, 'o-', label='max time = '+label)

def time_average(times, label): 

	#allentstimes =[]

	#for t in range(len(times)):  
	#	psit= exact_diag_evolve(psi, times[t], coeffs, L)
	#	entropies_na = []
	#	for na in range(1, L): 
	#		if na < L/2+1:
	#			rho = density_mat_half_line(psit, na, L)
	#			Sa = entropy(rho)		
	#			entropies_na.append(Sa)
	#		else: 
	#			rho = density_mat_half_line_left(psit, na, L)
	#			Sa = entropy(rho)		
	#			entropies_na.append(Sa)

	#	allentstimes.append(entropies_na)
		
	#with open('allentstimes_'+str(L)+'_Ystate_maxtime_'+label+'.npy', 'wb') as f:
	#	np.save(f,allentstimes)

	with open('allentstimes_'+str(L)+'_Ystate_maxtime_'+label+'.npy', 'rb') as f:
		allentstimes= np.load(f)
	
	#with open('allentstimes_'+str(L)+'_E_'+str(E)+'_delta_'+str(delta)+'_equal_sym.npy', 'rb') as f:
	#	allentstimes = np.load(f)

	renyi2 = plotvals(times, allentstimes)

	plt.plot(range(1,L), renyi2, 'o-', label='max time = '+label)
	



times = [times1a , times2, times3, times4]
labels = ['100', '150', '200', '250'] 

for i in range(len(times)): 
	time_average(times[i], labels[i])

with open('allentstimes_'+str(L)+'_Ystate_maxtime_'+'250'+'.npy', 'rb') as f:
	entropies_sofar=np.load(f)
 
time_average_append(times5, '350', entropies_sofar, plottimes5) 

with open('allentstimes_'+str(L)+'_Ystate_maxtime_'+'350'+'.npy', 'rb') as g:
	entropies_sofar2=np.load(g)

time_average_append(times6, '450', entropies_sofar2, plottimes6) 

#omega = np.zeros((2**L, 2**L))

#for i in inds: 
#	omega = omega +(1.0/num)*np.outer(eigvecs[:, i], np.conjugate(eigvecs[:, i])) 

#omega_prime = omega 

#for i in range(len(degenerate_pairs)): 
#	print 'here'
#	omega_prime = omega_prime + (1.0/num)*np.outer(eigvecs[:, degenerate_pairs[i][0]], np.conjugate(eigvecs[:, degenerate_pairs[i][1]])) 
#	+  (1.0/num)*np.outer(eigvecs[:, degenerate_pairs[i][1]], np.conjugate(eigvecs[:, degenerate_pairs[i][0]]))

#diags = []
#for na in range(1, L):
#	diags.append(diag_approx(omega, na, 2))

#with open('diags_'+str(L)+'_Ystate.npy', 'wb') as g:
#	np.save(g, diags)

with open('diags_'+str(L)+'_Ystate.npy', 'rb') as g:
	diags = np.load(g)
#with open('diags_'+str(L)+'_E_'+str(E)+'_delta_'+str(delta)+'_equal.npy', 'rb') as g:
#	diags = np.load(g)


#diags_prime = []
#for na in range(1, L):
#	diags_prime.append(diag_approx(omega_prime, na, 2))

#with open('diags_'+str(L)+'_E_'+str(E)+'_delta_'+str(delta)+'_equal_prime.npy', 'wb') as g:
#	np.save(g, diags_prime)

plt.plot(range(1,L), diags, '>-', label='diagonal approx')
#plt.plot(range(1,L), diags_prime, '<-', label='diagonal with degeneracies')
plt.xlabel('t')
plt.ylabel(r'$S_2$')
plt.legend()
plt.title('L='+str(L)+', Y state')
plt.show()

