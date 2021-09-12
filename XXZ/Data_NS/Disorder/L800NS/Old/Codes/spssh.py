# -*- coding: utf-8 -*-
"""
Calculation of the Loschmidt Echo for SSH Model with onsite disorder potential
using single-particle Hamiltonian and formula using determinant
Jesko Sirker
Ye Cheng (Lewis) Chen
"""
from __future__ import print_function
import numpy as np
import argparse


# parameters for entanglement calculations
parser = argparse.ArgumentParser(description='Calculation of Loschmidt echo for disordered fermions')
parser.add_argument('-L', type=int, default=800,
                    help='# system length L')
parser.add_argument('-W', type=float, default=0.001,
                    help='width box potential disorder')
parser.add_argument('-delta', type=float, default=-0.3,
                    help='# quench parameter delta')
parser.add_argument('-dt1', type=float, default=0.0008,
                    help='# discrete time interval first part')
parser.add_argument('-dt2', type=float, default=0.08,
                    help='# discrete time interval second part')
parser.add_argument('-tmid', type=float, default=2,
                    help='# maximum time range first part')
parser.add_argument('-tmax', type=float, default=10,
                    help='# maximum time range second part')
parser.add_argument('-dat', type=int, default=1,
                    help='# data size')
parser.add_argument('-sample', type=int, default=1,
                    help='# samples')
args=parser.parse_args()

# construct Hamiltonian for SSH model with disorder in diagonal elements
# diagnonal elements are random numbers in [-W/2,W/2]
def construct_DW(args):
    if args.W != 0.0:
        a = args.W * np.random.random_sample(args.L) - args.W/2        
    else:
        a = np.zeros(args.L)
    A = np.diag(a,0)
    return A

# construct single-particle Hamiltonian for SSH model
def construct_SPH(delta,L):
    H = np.zeros((L,L))
    for i in range(0,L-1):
        H[i,i+1]=1-delta*(-1)**i
        H[i+1,i]=1-delta*(-1)**i
    return H

# construct unitary time evolution operator Uexp(-iDt)U*
def construct_U(v,U,t):
    Ut = np.dot(U.conjugate(),np.dot(np.diag(np.exp(-1j*v*t)),(U.transpose()).conjugate()))
    return Ut

# construct two point correlation matrix for HD
def construct_CM(length):
    #m1=np.array([[1,0],[0,0]])
    #smat=np.kron(m1,np.identity(args.L/2))
    #CM = np.dot(U.transpose(),np.dot(smat,U))  # CM in eigebasis
    Neel = np.zeros((int(length),int(length/2)))
    for i in range(0,int(length)):
        for j in range(0,int(length/2)):
            if i+1==2*(j+1)-1: Neel[i,j]=1
    CM = np.dot(Neel,Neel.transpose())          # CM in Ising basis
    return CM

# calculate LE using |det(1-C+C*exp(-iHt))|
def calc_detLE(v,U,CM,t):
    LE=np.zeros(len(t))
    for i in t:
        Ut = construct_U(v,U,i)
        k=t.tolist().index(i)
        LE[k]=np.abs(np.linalg.det(np.subtract(np.identity(args.L),np.subtract(CM,np.dot(CM,Ut)))))
    return LE


# Run the program: time evolution of Neel state in single particle
samples=args.sample
t1=np.arange(0,args.tmid,args.dt1)
t2=np.arange(args.tmid,args.tmax+args.dt2,args.dt2)
t=np.concatenate((t1,t2), axis=0)
dat=np.zeros((args.dat,len(t)))
for i in range(0,args.dat):
    Sent=0
    for samp in range (0,samples):
        A = construct_DW(args)
        HF=A+construct_SPH(-args.delta,args.L)  # quench to -delta
        v,U = np.linalg.eigh(HF)
        CM=construct_CM(args.L)
        Sent+=calc_detLE(v,U,CM,t)
    dat[i]=(Sent/samples).transpose()


size=args.dat*(args.dat-1)/2
newdat=np.zeros((int(size),len(t)))
a=0
b=0
for i in dat:
    for j in range(a+1,len(dat)):
        newdat[b]=(i+dat[j])/2
        b=b+1
    a=a+1
LE=0
for i in range(0,int(size)):
    LE+=newdat[i]
if args.dat==1:
    result=dat[0]
elif args.dat > 1:
    result=np.divide(LE,size)

for item in result:
    print(item)


