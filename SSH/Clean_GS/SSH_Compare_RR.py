# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import scipy.special
import numpy as np
import argparse
import matplotlib.pyplot as plt
from bisect import bisect_left as findIndex
import time
import os
import psutil

# parameters for entanglement calculations
parser = argparse.ArgumentParser(description=
'Calculation of Loschmidt echo for disordered fermions')
parser.add_argument('-L', type=int, default=14,
                    help='# system length L')
parser.add_argument('-V', type=float, default=0.0,
                    help='V interaction term')
parser.add_argument('-W', type=float, default=0.0,
                    help='width box potential disorder')
parser.add_argument('-delta', type=float, default=-0.95,
                    help='# quench parameter delta')
parser.add_argument('-tint', type=float, default=0,
                    help='# initial time')
parser.add_argument('-tmax', type=float, default=20,
                    help='# maximum time range')
parser.add_argument('-dt', type=float, default=0.01,
                    help='# discrete time interval')
parser.add_argument('-dat', type=int, default=1,
                    help='# data size')
parser.add_argument('-sample', type=int, default=1,
                    help='# samples')
parser.add_argument('-openbc', type=int, default=0,
                    help='OBC = 1, PBC = 0')
args=parser.parse_args()


############################################################################
# Part 1
# Calculation for Loschmidt echo using single-particle Hamiltonian technique
# and formula |det(1-C+C*exp(-iHt))|
############################################################################
# construct Hamiltonian for SSH model with disorder in diagonal elements
# diagnonal elements are random numbers in [-W,W]
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
# Part 2
# Calculation for Loschmidt echo using many-particle Hamiltonian technique
# and formula |<exp(-iHt)>|
############################################################################
# State function configurations in spinless fermions
def manyPsi(particle,site):
    a=np.arange(2**site)
    bitcount=np.array([bin(x).count("1") for x in a])
    b=a.compress(bitcount==particle).tolist()[::-1]
    aList=[]
    for item in b:
        a=bin(int(item))[2:].zfill(site)
        aList.append(a)
    return aList

# C+C fermionic operators on Neel configurations
def cpc(l,j,Psi):
    Psi2=Psi.copy()
    for item in Psi2:
        k = Psi2.index(item)
        if item[j-1]=='0':
            Psi2[k]=list(item)
            Psi2[k][j-1]='1'
            Psi2[k]="".join(Psi2[k])
        if item[j-1]=='1':
            Psi2[k]=list(item)
            Psi2[k][j-1]='0'
            Psi2[k]="".join(Psi2[k])
    for item in Psi2:
        k = Psi2.index(item)
        if item[l-1]=='0':
            Psi2[k]=list(item)
            Psi2[k][l-1]='1'
            Psi2[k]="".join(Psi2[k])
        if item[l-1]=='1':
            Psi2[k]=list(item)
            Psi2[k][l-1]='0'
            Psi2[k]="".join(Psi2[k])
    return Psi2

def construct_MPDW(Psi,L,W):
    if args.W != 0.0:
        a = 2*W * np.random.random_sample(L) - W #mu in [-W,W]
    else:
        a = np.zeros(L)
    C = np.zeros(len(Psi))
    for i in range(len(Psi)):
        item=Psi[i]
        for j in range(len(item)):
            if item[j]=='1':
                C[i]+=a[j]
    A = np.diag(C)
    return A

def findMagnetizationStates(length,particlenumber):
    """
    constructs a table with the integer representations of all binaries
    with a given number of 1s
    """
    s = np.arange(2**length)
    bitcount = np.array([bin(x).count("1") for x in s])
    return s.compress(bitcount==particlenumber)

def bit(state,j,length):
    """return value of bit j"""
    return state >> (length-1-j) & 1

def bitFlip(state,j,k,length):
    """flip bits j and k of state if they are unequal"""
    return state ^ (2**(length-1-j)+2**(length-1-k))

# construct many-particle Hamiltonian for clean SSH model
def construct_MPH(V,delta,length,table):
    """construct clean Hamiltonian"""
    nos = len(table)
    h = np.zeros((nos,nos),np.float)
    for b,t in enumerate(table): # loop over eigenstates
        for j in range(length-args.openbc): # loop over sites
            k = (j+1)%length # right neighboring site
            # Heisenberg interaction for equal nearest neighbors
            if bit(t,j,length)==bit(t,k,length):
                h[b,b]+=0.25*V
            # Heisenberg interaction + tunneling for unequal nearest neighbors
            else:
                h[b,b]-=0.25*V
                bp = findIndex(table, bitFlip(t, j, k, length))
                #bp = findIndex(table, bitFlip(t, j, k, length))-1
                h[b,bp]=1.0-delta*(-1)**j
    return h


def calc_LE(vmi,Umi,vmf,Umf,t): # return rate starting from intial state
    """compute time evolution from ground state of Hi"""
    Uindex=vmi.tolist().index(min(vmi))
    Ui=Umi[Uindex]
    LE = np.zeros(len(t))
    evol = np.exp(-1j*t[:,np.newaxis]*vmf[np.newaxis])
    vv=np.dot(Umf.transpose().conj(),Ui)
    LE=np.abs(np.inner(evol*vv[np.newaxis,:],vv[np.newaxis,:]))
    return LE


############################################################################
# Part 3
# Run the program for both cases
############################################################################
t1=time.time()
t=np.arange(args.tint,args.tmax+args.dt/2,args.dt)


# calculate part 1
SPHi = construct_SPH(args.delta,args.L,args.openbc)
SPHf = construct_SPH(-args.delta,args.L,args.openbc)

# calculate part 2
particlenumber=args.L/2
Psi=manyPsi(particlenumber,args.L)
table = findMagnetizationStates(args.L,particlenumber)
MPDW = construct_MPDW(Psi,args.L,args.W)
MPHi = construct_MPH(args.V,args.delta,args.L,table)
MPHf = construct_MPH(args.V,-args.delta,args.L,table)
""" VERY IMPORTANT: do not use np.linalg.eigh here because it will run into problems """
vmi,Umi = np.linalg.eig(MPHi) 
Umi=Umi.transpose()
vmf,Umf = np.linalg.eig(MPHf) 


dat1=np.zeros((args.dat,len(t)))
dat2=np.zeros((args.dat,len(t)))
for i in range(args.dat):
    Store1=0
    Store2=0
    for samp in range(int(args.sample)):
        APDW = construct_APDW(args.L,args.W)
        MPDW = construct_MPDW(Psi,args.L,args.W)
        SPHiW = SPHi + APDW
        SPHfW = SPHf + APDW
        MPHiW = MPHi + MPDW
        MPHfW = MPHf + MPDW
        vsi,Usi = np.linalg.eigh(SPHiW)
        vsf,Usf = np.linalg.eigh(SPHfW)
        CM = construct_CM(Usi,args.L)
        vmi,Umi = np.linalg.eigh(MPHiW) 
        Umi=Umi.transpose()
        vmf,Umf = np.linalg.eigh(MPHfW) 
        Store1 += calc_detLE(vsf,Usf,CM,t)
        Store2 += calc_LE(vmi,Umi,vmf,Umf,t)
    dat1[i] += np.squeeze(Store1/args.sample)
    dat2[i] += np.squeeze(Store2/args.sample)

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

newdat2=np.zeros((int(size),len(t)))
a=0
b=0
# Averaging LE2
for k in dat2:
    for j in range(a+1,len(dat2)):
        newdat2[b]=np.divide(k+dat2[j],2.0)
        b=b+1
    a=a+1
LE2=0
for k in range(0,int(size)):
    LE2+=newdat2[k]
if args.dat==1:
    result2=dat2[0]
elif args.dat > 1:
    result2=np.divide(LE2,size)


LE1=result1
LE2=result2
RR1=-2*np.log(LE1)/args.L
RR2=-2*np.log(LE2)/args.L

""" Plot the results """
if args.openbc == 1:
    BC = 'OBC'
elif args.openbc == 0:
    BC = 'PBC'
tiTle1='-->-'+str(args.delta)+' and '+str(BC)
if args.delta < 0:
    tiTle1='-->+'+str(np.abs(args.delta))+' and '+str(BC)
plt.plot(t,LE1,'r',label='SPH Approach')
plt.plot(t,LE2,'b--',label='MPH Approach')
plt.legend(loc='upper right');
plt.ylabel('$\mathcal{L}(t)$')
plt.xlabel('t')
if args.delta < 0:
    tiTleLE2='Loschmidt echo for SSH with L = '+str(args.L)+', $\delta$ = '+str(args.delta)
elif args.delta > 0:
    tiTleLE2='Loschmidt echo for SSH with L = '+str(args.L)+', $\delta$ = +'+str(args.delta)
tiTleL=tiTleLE2+tiTle1
plt.title(tiTleL,fontsize=10.5)
plt.show()
plt.plot(t,RR1,'r',label='SPH Approach')
plt.plot(t,RR2,'b--',label='MPH Approach')
plt.legend(loc='upper right');
plt.ylabel('l(t)')
plt.xlabel('t')
if args.delta < 0:
    tiTleRR2='Return rate for SSH with L = '+str(args.L)+', $\delta$ = '+str(args.delta)
elif args.delta > 0:
    tiTleRR2='Return rate for SSH with L = '+str(args.L)+', $\delta$ = +'+str(args.delta)
tiTleR=tiTleRR2+tiTle1
plt.title(tiTleR,fontsize=10.5)
plt.show()

fileLE1a='SSH'+str(BC)+'_SPH_'+str(args.L)+'LE_del='
fileLE1b=str(args.delta)+'_W='+str(args.W)+'.dat'
fileLE1 =fileLE1a+fileLE1b
f=open(fileLE1,'w')
for item in LE1:
    print(item,file=f)
f.close()
fileLE2a='SSH'+str(BC)+'_MPH_'+str(args.L)+'LE_del='
fileLE2b=str(args.delta)+'_W='+str(args.W)+'.dat'
fileLE2 =fileLE2a+fileLE2b
g=open(fileLE2,'w')
for item in LE2:
    print(item,file=g)
g.close()

t2=time.time()
print('Total time is ',(t2-t1)/60,' minutes.')
process = psutil.Process(os.getpid())
print(process.memory_info().rss)
print(str(args))
    