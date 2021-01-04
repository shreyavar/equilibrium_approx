#!/usr/bin/env python
import numpy as np
import scipy
from numpy.linalg import eigh
import itertools
import cmath 

X = [[0.0, 1.0],[1.0, 0.0]]
Y = [[0.0, -1.0j],[1.0j, 0.0]]
Z = [[1.0, 0.0],[0.0, -1.0]]
I = [[1.0, 0.0],[0.0, 1.0]]

L=11
h = 0.5
g = -1.05


def multikron(A):
	prod =1
	for i in range(len(A)):
		prod = np.kron(prod, A[i])
	return prod

def complexify(X): 
	return np.add(X, 1j*np.zeros(X.shape))


def diagonalize(L):
	term1 = np.zeros((2**L, 2**L))
	term2 = np.zeros((2**L, 2**L))
	term3 = np.zeros((2**L, 2**L))

	for i in range(L-1):
		term1array = []
		term2array = []
		term3array = []
		for j in range(L): 
			if j== i:
				term1array.append(Z)
				term2array.append(Z)
				term3array.append(X)
			elif j == i+1:
				term1array.append(Z)
				term2array.append(I)
				term3array.append(I)
			else:
				term1array.append(I)
				term2array.append(I)
				term3array.append(I)
		term1 = np.add(term1, multikron(term1array))
		term2 = np.add(term2, multikron(term2array))
		term3 = np.add(term3, multikron(term3array))

	finalterm1array =[]
	finalterm2array =[]
	finalterm3array =[]
	for j in range(L):
		if j == L-1: 
			finalterm1array.append(Z)
			finalterm2array.append(Z)
			finalterm3array.append(X)
		elif j==0: 
			finalterm1array.append(Z)
			finalterm2array.append(I)
			finalterm3array.append(I)
		else:
			finalterm1array.append(I)
			finalterm2array.append(I)
			finalterm3array.append(I)
	term1 = np.add(term1, multikron(finalterm1array))
	term2 = np.add(term2, multikron(finalterm2array))
	term3 = np.add(term3, multikron(finalterm3array))
	
	Ham = np.add(term1, h*term2)
	Ham = np.add(Ham, g*term3)

	vals,vecs = eigh(Ham)
	vals = complexify(vals)
	vecs = complexify(vecs)

	print [vals[0], vals[1], vals[2], vals[3]]
	print vals[0]/L

	np.savetxt('chaoticvalsnew'+str(L),vals.view(float))
	np.savetxt('chaoticvecsnew'+str(L),vecs.view(float))
	#eigvecs = np.loadtxt('chaoticvecsnew'+str(L)).view(complex)
	#eigvals = np.loadtxt('chaoticvalsnew'+str(L)).view(complex)
	#print eigvecs.shape
	#print eigvals.shape

diagonalize(L)

