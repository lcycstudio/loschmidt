# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import scipy.special
import numpy as np
import argparse


# parameters for entanglement calculations
parser = argparse.ArgumentParser(description=
'Calculation of Loschmidt echo for disordered fermions')
parser.add_argument('-L', type=int, default=50,
                    help='# system length L')
parser.add_argument('-W', type=float, default=0.0,
                    help='width box potential disorder')
parser.add_argument('-delta', type=float, default=0.3,
                    help='# quench parameter delta')
parser.add_argument('-dt1', type=float, default=0.0008,
                    help='# discrete time interval first part')
parser.add_argument('-dt2', type=float, default=0.05,
                    help='# discrete time interval second part')
parser.add_argument('-tint', type=float, default=0,
                    help='# maximum time range first part')
parser.add_argument('-tmid', type=float, default=2,
                    help='# maximum time range first part')
parser.add_argument('-tmax', type=float, default=20,
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
def construct_SPH(delta,L,openbc):
    H = np.zeros((L,L))
    for i in range(0,L-1):
        H[i,i+1]=1.0-delta*(-1)**i
        H[i+1,i]=1.0-delta*(-1)**i
    if openbc == 0:
        H[0][-1]=1.0+delta*(-1)**L
        H[-1][0]=1.0+delta*(-1)**L
    return H

# construct unitary time evolution operator Uexp(-iDt)U*
def construct_U(v,U,t):
    Ut = np.dot(U.conjugate(),np.dot(np.diag(np.exp(-1j*v*t)),(U.transpose()).conjugate()))
    return Ut

# construct two point correlation matrix for HD
def construct_CM(U,L):
    m1=np.array([[1,0],[0,0]])
    smat=np.kron(m1,np.identity(int(L/2)))
    CM = np.dot(U,np.dot(smat,U.transpose()))
    return CM

# calculate LE using |det(1-C+C*exp(-iHt))|
def calc_detLE(v,U,CM,t):
    LE=np.zeros(len(t))
    for i in t:
        Ut = construct_U(v,U,i)
        k=t.tolist().index(i)
        LE[k]=np.abs(np.linalg.det(np.identity(args.L)-CM+np.dot(CM,Ut)))
    return LE


############################################################################
# Part 3
# Run the program for both cases
############################################################################
t1=np.arange(args.tint,args.tmid,args.dt1)
t2=np.arange(args.tmid,args.tmax+args.dt2,args.dt2)
t=np.concatenate((t1,t2), axis=0)

# calculate part 1
SPHi = construct_SPH(args.delta,args.L,args.openbc)
SPHf = construct_SPH(-args.delta,args.L,args.openbc)

dat1=np.zeros((args.dat,len(t)))
for i in range(args.dat):
    Store1=0
    for samp in range(int(args.sample)):
        APDW = construct_APDW(args.L,args.W)
        SPHiW = SPHi + APDW
        SPHfW = SPHf + APDW
        vsi,Usi = np.linalg.eigh(SPHiW)
        vsf,Usf = np.linalg.eigh(SPHfW)
        CM = construct_CM(Usi,args.L)
        Store1 += calc_detLE(vsf,Usf,CM,t)
    dat1[i] += np.squeeze(Store1/args.sample)

size=np.divide(args.dat*(args.dat-1),2.0)
newdat1=np.zeros((int(size),len(t)))
a=0
b=0
# Averaging LE2
for k in dat1:
    for j in range(a+1,len(dat1)):
        newdat1[b]=np.divide(k+dat1[j],2.0)
        b=b+1
    a=a+1
LE1=0
for k in range(0,int(size)):
    LE1+=newdat1[k]
if args.dat==1:
    result1=dat1[0]
elif args.dat > 1:
    result1=np.divide(LE1,size)

LE1=result1
RR1=-2*np.log(LE1)/args.L

for item in RR1:
    print(item)
    