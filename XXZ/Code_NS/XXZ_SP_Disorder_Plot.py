# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import scipy.special
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
import os
import psutil
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)



# parameters for entanglement calculations
parser = argparse.ArgumentParser(description=
'Calculation of Loschmidt echo for disordered fermions')
parser.add_argument('-L', type=int, default=6,
                    help='# system length L')
parser.add_argument('-W', type=float, default=0.0,
                    help='width box potential disorder')
parser.add_argument('-dt1', type=float, default=0.01,
                    help='# discrete time interval first part')
parser.add_argument('-dt2', type=float, default=0.01,
                    help='# discrete time interval second part')
parser.add_argument('-tint', type=float, default=0,
                    help='# maximum time range first part')
parser.add_argument('-tmid', type=float, default=3,
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
def construct_SPH(L,openbc,t):
    H = np.zeros((L,L))
    for i in range(0,L-1):
        H[i,i+1]=t*0.5
        H[i+1,i]=t*0.5
    if openbc == 0:
        H[0][-1]=t*0.5
        H[-1][0]=t*0.5
    return H

# construct unitary time evolution operator Uexp(-iDt)U*
def construct_U(v,U,t):
    Ut = np.dot(U.conjugate(),np.dot(np.diag(np.exp(-1j*v*t)),(U.transpose()).conjugate()))
    return Ut

# construct two point correlation matrix for HD
def construct_CM(L):
    Neel = np.zeros((int(L),int(L/2)))
    for i in range(0,int(L)):
        for j in range(0,int(L/2)):
            if i+1==2*(j+1)-1: Neel[i,j]=1
    CM = np.dot(Neel,Neel.transpose())          # CM in Ising basis
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
tic1=time.time()
t1=np.arange(args.tint,args.tmid,args.dt1)
t2=np.arange(args.tmid,args.tmax+args.dt2,args.dt2)
t=np.concatenate((t1,t2), axis=0)


# calculate part 1
tt=1.0
SPH = construct_SPH(args.L,args.openbc,tt)
vs,Us = np.linalg.eigh(SPH)
CM = construct_CM(args.L)

dat1=np.zeros((args.dat,len(t)))
for i in range(args.dat):
    Store1=0
    Store2=0
    for samp in range(int(args.sample)):
        APDW = construct_APDW(args.L,args.W)
        SPHfW = SPH + APDW
        vsf,Usf = np.linalg.eigh(SPHfW)
        Store1 += calc_detLE(vsf,Usf,CM,t)
    dat1[i] += np.squeeze(Store1/args.sample)


size=np.divide(args.dat*(args.dat-1),2.0)
newdat1=np.zeros((int(size),len(t)))
a=0
b=0
# Averaging LE1
for i in dat1:
    for j in range(a+1,len(dat1)):
        newdat1[b]=np.divide(i+dat1[j],2.0)
        b=b+1
    a=a+1
LE1=0
for i in range(0,int(size)):
    LE1+=newdat1[i]
if args.dat==1:
    result1=dat1[0]
elif args.dat > 1:
    result1=np.divide(LE1,size)

    
LE1=result1
RR1=-2*np.log(LE1)/args.L
""" Plot the results """
if args.openbc == 1:
    BC = 'OBC'
elif args.openbc == 0:
    BC = 'PBC'
tiTle1=' and '+str(BC)
plt.plot(t,LE1,'r',label='SPH Approach')
plt.legend(loc='upper right');
plt.ylabel('$\mathcal{L}(t)$')
plt.xlabel('t')
tiTleLE2='Loschmidt echo for XXZ with L = '+str(args.L)
tiTleL=tiTleLE2+tiTle1
plt.title(tiTleL,fontsize=10.5)
plt.show()
plt.plot(t,RR1,'r',label='SPH Approach')
plt.legend(loc='upper right');
plt.ylabel('l(t)')
plt.xlabel('t')
tiTleRR2='Return rate for XXZ with L = '+str(args.L)
tiTleR=tiTleRR2+tiTle1
plt.title(tiTleR,fontsize=10.5)
plt.axvline(x=np.pi/2,color=colors['grey'],linestyle='--')
plt.show()

fileLE1a='XXZ_'+str(BC)+'_SPH_'+str(args.L)+'LE_del='
fileLE1b='_W='+str(args.W)+'.dat'
fileLE1 =fileLE1a+fileLE1b
f=open(fileLE1,'w')
for item in LE1:
    print(item,file=f)
f.close()

tic2=time.time()
print('Total time is ',(tic2-tic1)/60,' minutes.')
process = psutil.Process(os.getpid())
print(process.memory_info().rss)
print(str(args))

"""
ka=-np.pi/2
kb=np.pi/2
N=10000
dk=(kb-ka)/N
dkw=np.arange(-np.pi/2,np.pi/2+dk,dk)
k=ka
tti=np.arange(0,20,0.01)
res=np.zeros(len(tti))
for ii in range(len(tti)):
    jj=tti[ii]
    kint=0
    for item in dkw:
        kint=kint+dk*np.log((np.cos(jj*np.cos(item + 0.5*dk)))**2)
    res[ii]=-kint/(2*np.pi)

plt.subplot(2,1,1)
plt.plot(t,RR1,label=r'L='+str(args.L))
plt.legend(loc='upper right');
plt.ylabel('l(t)')
plt.xlabel('t')
plt.yticks([0,1])
plt.xticks([0,np.pi/2,5,10,15,20],[0,r'$\pi/2$',5,10,15,20])
plt.axvline(x=np.pi/2,color=colors['grey'],linestyle='--')
plt.text(0.6, 0.5, "(A)")
plt.show()

plt.subplot(2,1,2)
plt.plot(tti,res,label=r'L$\rightarrow\infty$')
plt.legend(loc='upper right');
plt.ylabel('l(t)')
plt.xlabel('t')
plt.yticks([0,1])
plt.xticks([0,np.pi/2,5,10,15,20],[0,r'$\pi/2$',5,10,15,20])
plt.axvline(x=np.pi/2,color=colors['grey'],linestyle='--')
plt.show()
"""

