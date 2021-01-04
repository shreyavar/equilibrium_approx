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
Lmax = 12
eigvecs = np.loadtxt('chaoticvecsnew'+str(L)).view(complex)
eigvals = np.real(np.loadtxt('chaoticvalsnew'+str(L)).view(complex))

times1 = np.arange(10.0, 40.0, 10.0)
times2 = np.arange(10.0, 155.0, 5.0)
times2s = np.arange(15.0, 165.0, 5.0)
times3 = np.arange(10.0, 152.0, 2.0)

#basic functions 

def complexify(X): 
	return np.add(X, 1j*np.zeros(X.shape))

def multikron(A):
	prod =1
	for i in range(len(A)):
		prod = np.kron(prod, A[i])
	return prod


#evolving with ED 

def exact_diag_evolve(psi, t, coeffs, eigvecs, eigvals, L): 
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
	for i in range(len(p)):
		#print p[i] 
		if p[i]!=0:
			for j in range(6): 
				if j==0: 
					ent[j] = ent[j]- p[i]*np.log(p[i])
				else: 
					ent[j] = ent[j] + p[i]**(j+1)
	for j in range(1, 6): 
		ent[j] = -1.0/(j)*np.log(ent[j])

	return ent


#average energy for a given beta

def av_en(beta): 
	en = 0.0
	z = 0.0
	for i in range(2**L): 
		en = en + np.exp(-beta*np.real(eigvals[i]))*np.real(eigvals[i])
		z = z + np.exp(-beta*np.real(eigvals[i]))
	return en/z

#average energy of a state, given its coefficients along the energy eigenstates 
def av_en_coeff(coeffs): 
	en = 0.0 
	en2 = 0.0
	for i in range(2**L): 
		en = en + (np.abs(coeffs[i])**2)*eigvals[i]
		en2 = en2 + (np.abs(coeffs[i])**2)*((eigvals[i])**2)
	return [np.real(en), np.real(en2)]


#create initial product state  

theta = np.random.uniform(0, np.pi)
phi = np.random.uniform(0, 2*np.pi)

str0 = 'theta='+str(theta)+', phi='+str(phi)

psi_list = []
for i in range(L): 
	psi_list.append([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
psi=multikron(psi_list) 


#find coefficients of initial state along all energy eigenstates 

coeffs = []
for i in range(2**L): 
	coeff= np.dot(np.conjugate(eigvecs[:,i]), psi)
	coeffs.append(coeff)


#average energy of the initial state 

en_ev = av_en_coeff(coeffs)
str1 = '. The expectation value of the energy in the initial state is '+str(en_ev[0]) 
str2 = ' and the energy variance in the initial state is '+str(en_ev[1])


#solve for beta associated with initial state 
func = lambda bet : en_ev[0]-av_en(bet)
beta_initial_guess = 1.0
beta_state = fsolve(func, beta_initial_guess)
str3= ' and the temperature associated with the average energy of the initial state is '+str(beta_state)

Lines = [ str0, str1, str2, str3 ] 
file1 = open('info_theta_'+str(int(theta*100)*0.01)+'_phi_'+str(int(phi*100)*0.01)+'.txt',"w") 
file1.writelines(Lines) 
file1.close() 

#diagonal approximation for second Renyi entropy 

def canonical_ent_diag_approx_unfactorized_2(na): 	
	ent = 0 
	
	omega = np.zeros((2**L, 2**L))

	for i in range(2**L):
		omega = omega + (np.abs(coeffs[i])**2)*np.outer(eigvecs[:, i], np.conjugate(eigvecs[:, i])) 

	omega_a = partial_trace(omega, L, na)
	omega_b = partial_trace_left(omega, L, L-na)

	num = np.trace(np.matmul(omega_a, omega_a))+ np.trace(np.matmul(omega_b, omega_b))

	return  -1.0*np.log(num)


def diag_approx_vals(): 
	diagonal_sa = []
	for na in range(1, L):  
		diagonal_sa.append(canonical_ent_diag_approx_unfactorized_2(na))
	diagonal = np.array(diagonal_sa)
	np.savetxt('diagonal_L_'+str(L)+'_theta_'+str(int(theta*100)*0.01)+'_phi_'+str(int(phi*100)*0.01), diagonal.view(float))
	#diagonal_sa = np.loadtxt('diagonal_L_'+str(L)+'_theta_'+str(int(theta*100)*0.01)+'_phi_'+str(int(phi*100)*0.01)).view(complex)
	return diagonal_sa


# Permutation group manipulations 

def count(A, b):
	count = 0  
	for i in range(len(A)): 
		for j in range(len(A[i])): 
			if A[i][j]==b: 
				count = count +1 

	return count

def cyclenums(A): 
	allcyclenum = []
	for i in range(len(A)): 
		allcyclenum.append(len(A[i]))
		
	return allcyclenum

def blocklengths(A): 
	alllengths = []
	for i in range(len(A)): 
		lengths =[]
		for j in range(len(A[i])): 
			lengths.append(len(A[i][j]))
		alllengths.append(lengths)
	return alllengths

def fillcycle(permcycles, n):
	newcycles = []
	for i in range(len(permcycles)):
		newcycle = permcycles[i]
		for k in range(n): 
			if count(permcycles[i], k)==0: 
				newcycle.append([k])
		newcycles.append(newcycle)

	return newcycles

def permutation_info(n): 
	G = SymmetricGroup(n); 
	tau = list(G.generate_schreier_sims(af=True))
	permcycles = [list(Permutation(tau[i]).cyclic_form) for i in range(len(tau))]
	taucycles = fillcycle(permcycles, n)
	etain = Permutation([[i for i in range(n)]])
	tauetain = [list((tau[i]*etain).cyclic_form) for i in range(len(tau))]
	etaincycles = fillcycle(tauetain, n)

	return [blocklengths(taucycles), blocklengths(etaincycles)] 


def permutation_info_mc(n): 
	G = SymmetricGroup(n); 
	tau = list(G.generate_schreier_sims(af=True))
	permcycles = [list(Permutation(tau[i]).cyclic_form) for i in range(len(tau))]
	taucycles = fillcycle(permcycles, n)
	etain = Permutation([[i for i in range(n)]])
	tauetain = [list((tau[i]*etain).cyclic_form) for i in range(len(tau))]
	etaincycles = fillcycle(tauetain, n)

	return [cyclenums(taucycles), cyclenums(etaincycles)] 


#canonical equilibrium approximation 

def gb(b, L): 
	eigvals = np.loadtxt('chaoticvalsnew'+str(L)).view(complex)
	gb = 0.0
	for i in range(len(eigvals)):
		gb = gb + np.exp(-b*eigvals[i])

	return np.log(gb)/L


betas = np.arange(-10.0, 10.0, 0.01)
gb = [ np.real(gb(betas[i], Lmax)) for i in range(len(betas))]
np.savetxt('gb_data', gb)
#gb = np.loadtxt('gb_data')
gb_interp = interp1d(betas, gb)


def zc(beta, na):
	return np.exp(na*gb_interp(beta)) 


def canonical_eq_approx(beta, na, n):
	ent = 0 
	den = zc(beta, L)**n
	[blocks1, blocks2] = permutation_info(n)
	permsum = 0.0
	for i in range(math.factorial(n)):
		prod = 1 
		for j in range(len(blocks1[i])): 
			prod = prod*zc(blocks1[i][j]*beta, L-na)
		for k in range(len(blocks2[i])): 
			prod = prod*zc(blocks2[i][k]*beta, na)
		permsum = permsum + prod 

	return  -1.0/(n-1)*np.log(permsum/den)



def canonical_approx_vals(beta, n): 

	canonical_sa = []
	for na in range(1, L):  
		canonical_sa.append(canonical_eq_approx(beta, na, n))
	canonical = np.array(canonical_sa)
	np.savetxt('canonical_L_'+str(L)+'_theta_'+str(int(theta*100)*0.01)+'_phi_'+str(int(phi*100)*0.01), canonical.view(float))
	#canonical_sa = np.loadtxt('canonical_L_'+str(L)+'_theta_'+str(int(theta*100)*0.01)+'_phi_'+str(int(phi*100)*0.01)).view(complex)
	return canonical_sa


#microcanonical equilibrium approximation 

def ent_dens(): 
	eigvals = np.loadtxt('chaoticvalsnew'+str(Lmax)).view(complex)
	
	min_e = round(round(eigvals[0]/Lmax,1)-0.1,1)
	max_e = round(round(eigvals[-1]/Lmax,1)+0.1,1)

	energy_densities = np.array(range(int(min_e*10), int(max_e*10)))*0.1
	#print energy_densities

	g_e = []
	j =0
	for i in range(len(energy_densities)-1): 
		
		dens = 0
		while eigvals[j]<= L*energy_densities[i+1] and j<len(eigvals):
			#print 'start at:'+str(bins[i])
			#print eigvals[j]
			dens = dens+1
			if j< len(eigvals)-1:
				j=j+1
			else:
				break
		
		g_e.append(dens)
		
	np.savetxt('entropy_density_from_L_'+str(Lmax),[energy_densities[0:-1], g_e])
	#[energy_densities, g_e] = np.loadtxt('entropy_density_from_L_'+str(Lmax))
	return [energy_densities[0:-1], g_e]

def find_density(a):
	[en_dens, g_e] = ent_dens(); 
	#print en_dens
	#print g_e
	min_e = en_dens[0]*a
	max_e = en_dens[-1]*a
	energies = np.array(range(int(min_e*10), int(max_e*10)))*0.1

	#print len(en_dens)
	#print len(s) 

	f = interp1d(en_dens, g_e)
	
	#print eigvals
	densities = []
	norm = 0.0
	for i in range(len(energies)):
		weight = np.power(f(energies[i]/a),a*1.0/L)
		densities.append(weight)
		norm = norm + weight

	
	return [energies, np.array(densities)*(2**a)/norm]

def nearly_equal(a,b,sig_fig=1):
    return ( a==b or 
             int(a*10**sig_fig) == int(b*10**sig_fig)
           )

def microcanonical(n, na, E):
	
	energypairs =[]
	densitypairs =[]
	[energies_a, densities_a] = find_density(na)
	[energies_b, densities_b] = find_density(L-na)

	if na==3: 
		print E
		print 'na='+str(na)
		print "energy_a"
		print energies_a
		print "energy_b"
		print energies_b

	for i in range(len(energies_a)): 
		for j in range(len(energies_b)): 
			#if round(energies_b[j],1)+round(energies_a[i],1)==round(E,1): 
			if nearly_equal(round(energies_b[j]+energies_a[i],1), round(E,1)):
				energypairs.append([round(energies_a[i],1), round(energies_b[j],1)])
				densitypairs.append([densities_a[i], densities_b[j]])

	if na==3:
		print "energy pairs"
		print energypairs

	num = 0.0
	den = 0.0
	[cyclenum1, cyclenum2] = permutation_info_mc(n)

	for k in range(len(energypairs)):
		den = den + densitypairs[k][0]*densitypairs[k][1]
		permsum = 0 
		for l in range(len(cyclenum1)):
			permsum = permsum + (densitypairs[k][0]**cyclenum2[l])*(densitypairs[k][1]**cyclenum1[l])
		num = num + permsum 

	return -1.0/(n-1)*np.log(num/den**n)


def mic_approx_vals(E, n): 
	mic_sa = []
	for na in range(1, L):  
		mic_sa.append(microcanonical(n, na, E))
	mic = np.array(mic_sa)
	np.savetxt('microcanonical_L_'+str(L)+'_theta_'+str(int(theta*100)*0.01)+'_phi_'+str(int(phi*100)*0.01), mic.view(float))
	return mic_sa


#time-evolved entanglement entropies for a given range of times for all subsystem sizes. 
def entropies_times(times, label):	
	allentstimes =[]

	for t in range(len(times)): 
		time = times[t]
		psit= exact_diag_evolve(psi, time, coeffs, eigvecs, eigvals, L)

		allents = []
		
		for na in range(1, L): 
			
			rho = density_mat_half_line(psit, na, L)
			Sa = entropy(rho)		
			allents.append(Sa)
			
		allentstimes.append(allents)

	with open('allentstimes_'+str(L)+'_theta_'+str(int(theta*100)*0.01)+'_phi_'+str(int(phi*100)*0.01)+'_timerange'+label+'.npy', 'wb') as f:
		np.save(f,allentstimes)

	#with open('allentstimes_'+str(L)+'_theta_'+str(int(theta*100)*0.01)+'_phi_'+str(int(phi*100)*0.01)+'_timerange'+label+'.npy', 'rb') as f:
		allentstimes = np.load(f)

	return allentstimes 



def plotvals(times, label, allentstimes):

	plotvals =[]
	for na in range(1, L):
		timesarray= []
		for t in range(len(times)):
			timesarray.append(allentstimes[t][na-1][1])
		plotvals.append(np.mean(timesarray,axis=0))
	np.savetxt('av_ent_'+str(L)+'_theta_'+str(int(theta*100)*0.01)+'_phi_'+str(int(phi*100)*0.01)+'_n_'+str(2)+'timerange'+label, plotvals)

	return plotvals 


alltimes = [times1, times2 , times2s, times3]
labels = ['15', '28', '28a', '70']

for k in range(len(alltimes)):
	plt.plot(range(1, L), plotvals(alltimes[k], labels[k], entropies_times(alltimes[k], labels[k])), 'o-', label='average over '+labels[k])


plt.plot(range(1, L), canonical_approx_vals(beta_state, 2),'^-', label='canonical')
plt.plot(range(1, L), mic_approx_vals(en_ev[0], 2),'^-', label='microcanonical')
plt.plot(range(1, L), diag_approx_vals(),'^-', label='diagonal')


plt.title('L='+str(L)+', n='+str(2)+', theta='+str(int(theta*100)*0.01)+', phi='+str(int(phi*100)*0.01))
plt.legend(loc='best')
plt.xlabel(r'$N_{A}$')
plt.ylabel(r'$S_n$')
plt.show()




