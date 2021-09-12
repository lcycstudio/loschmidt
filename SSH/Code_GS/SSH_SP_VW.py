# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 17:40:44 2018

@author: LewisCYC
"""

#import scipy.special
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
import os
import psutil

# parameters for entanglement calculations
parser = argparse.ArgumentParser(description=
'Calculation of Loschmidt echo for disordered fermions')
parser.add_argument('-L', type=int, default=4,
                    help='# system length L')
parser.add_argument('-V', type=float, default=0.0,
                    help='V interaction term')
parser.add_argument('-W', type=float, default=0.0,
                    help='width box potential disorder')
parser.add_argument('-delta', type=float, default=3,
                    help='# quench parameter delta')
parser.add_argument('-dt1', type=float, default=0.1,
                    help='# discrete time interval first part')
parser.add_argument('-dt2', type=float, default=0.1,
                    help='# discrete time interval second part')
parser.add_argument('-tint', type=float, default=0,
                    help='# maximum time range first part')
parser.add_argument('-tmid', type=float, default=2,
                    help='# maximum time range first part')
parser.add_argument('-tmax', type=float, default=100,
                    help='# maximum time range second part')
parser.add_argument('-dat', type=int, default=1,
                    help='# data size')
parser.add_argument('-sample', type=int, default=1,
                    help='# samples')
parser.add_argument('-openbc', type=int, default=1,
                    help='OBC = 1, PBC = 0')
args=parser.parse_args()


############################################################################
# Part 1
# Calculation for Loschmidt echo using single-particle Hamiltonian technique
# and formula |det(1-C+C*exp(-iHt))|
############################################################################
# construct Hamiltonian for SSH model with disorder in diagonal elements
# diagnonal elements are random numbers in [-W/2,W/2]
def construct_APDW(L,W):
    if args.W != 0.0:
        a = 2*W * np.random.random_sample(L) - W
    else:
        a = np.zeros(L)
    A = np.diag(a,0)
    return A

# construct single-particle Hamiltonian for SSH model
def construct_SPHw(delta,L,openbc):
    H = np.zeros((L,L))
    if delta == 1:
        for i in range(0,L-1,2):
            H[i,i+1]=1.0
            H[i+1,i]=1.0
    elif delta == -1: #Wdelta=1
        for i in range(1,L-1,2):
            H[i,i+1]=1.0
            H[i+1,i]=1.0
            if openbc == 0:
                H[0][-1]=1.0
                H[-1][0]=1.0
    else:
        for i in range(1,L-1,2):
            H[i,i+1]=delta
            H[i+1,i]=delta
    return H

def construct_SPHv(delta,L,openbc):
    H = np.zeros((L,L))
    if delta == 1:
        for i in range(0,L-1,2):
            H[i,i+1]=1.0
            H[i+1,i]=1.0
    elif delta == -1: #Wdelta=1
        for i in range(1,L-1,2):
            H[i,i+1]=1.0
            H[i+1,i]=1.0
            if openbc == 0:
                H[0][-1]=1.0
                H[-1][0]=1.0
    else:
        for i in range(0,L-1,2):
            H[i,i+1]=delta
            H[i+1,i]=delta
    return H


""" For v=1, w=0 to 3"""
Vdelta=1
Wgroup=np.arange(0,args.delta+0.01,0.01)
ww=np.zeros(args.L)
Emat=np.zeros((len(Wgroup),args.L))
a=0
for i in Wgroup:
    APDW = construct_APDW(args.L,args.W)
    SPHv = construct_SPHw(Vdelta,args.L,args.openbc)+APDW
    SPHw = SPHv+construct_SPHw(i,args.L,args.openbc)+APDW
    vsv,Usv = np.linalg.eigh(SPHv)
    vsf,Usf = np.linalg.eigh(SPHw)
    Emat[a]=vsf
    a=a+1

SPHw = SPHv+construct_SPHw(0,args.L,args.openbc)
vsf,Usf = np.linalg.eigh(SPHw)
ww[:]=0
plt.plot(ww,vsf,'ro')
ET=Emat.transpose()
plt.plot()
for i in range(args.L):
    plt.plot(Wgroup,ET[i],'b')
plt.xticks([0,1,2,3])
plt.ylabel('Energy $E$')
plt.xlabel('$w$')
plt.ylim([-3,3])
plt.xlim([0,3])
plt.text(0.1, 2.5, r'$v=1$')
plt.show()


""" For w=1, v= 0 to 3"""
Wdelta=-1
vgroup=np.arange(0,args.delta+0.01,0.01)
vv=np.zeros(args.L)
Emat=np.zeros((len(vgroup),args.L))
a=0
for i in vgroup:
    APDW = construct_APDW(args.L,args.W)    
    SPHi = construct_SPHv(Wdelta,args.L,args.openbc)+APDW
    SPHf = SPHi+construct_SPHv(i,args.L,args.openbc)+APDW
    vsi,Usi = np.linalg.eigh(SPHi)
    vsf,Usf = np.linalg.eigh(SPHf)
    Emat[a]=vsf
    a=a+1

SPHf = SPHi+construct_SPHv(0,args.L,args.openbc)
vsf,Usf = np.linalg.eigh(SPHf)
vv[:]=0
plt.plot(vv,vsf,'ro')
ET=Emat.transpose()
plt.plot()
for i in range(args.L):
    plt.plot(vgroup,ET[i],'b')
plt.xticks([0,1,2,3])
plt.ylabel('Energy $E$')
plt.xlabel('$v$')
plt.ylim([-3,3])
plt.xlim([0,3])
plt.text(0.1, 2.5,  r'$w=1$')
plt.show()