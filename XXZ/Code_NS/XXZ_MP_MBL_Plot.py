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
parser.add_argument('-L', type=int, default=8,
                    help='# system length L')
parser.add_argument('-V', type=float, default=1.0,
                    help='V interaction term')
parser.add_argument('-W', type=float, default=6.5,
                    help='width box potential disorder')
parser.add_argument('-tint', type=float, default=0,
                    help='# initial time')
parser.add_argument('-tmid1', type=float, default=10**0,
                    help='# middle time 1')
parser.add_argument('-tmid2', type=float, default=10**2,
                    help='# middle time 2')
parser.add_argument('-tmid3', type=float, default=10**4,
                    help='# middle time 3')
parser.add_argument('-tmid4', type=float, default=10**5,
                    help='# middle time 4')
parser.add_argument('-tmid5', type=float, default=10**6,
                    help=' middle time 5')
parser.add_argument('-tmid6', type=float, default=10**8,
                    help='# middle time 6')
parser.add_argument('-tmid7', type=float, default=10**10,
                    help='# middle time 7')
parser.add_argument('-tmax1', type=float, default=10**12,
                    help='# maximum time range')
parser.add_argument('-tnum', type=float, default=200,
                    help='# discrete time number')
parser.add_argument('-dat', type=int, default=20,
                    help='# data size')
parser.add_argument('-sample', type=int, default=50,
                    help='# samples')
parser.add_argument('-openbc', type=int, default=0,
                    help='OBC = 1, PBC = 0')
args=parser.parse_args()



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
def construct_MPH(V,length,table):
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
                h[b,bp]=0.5
    return h

# find Neel state in eigenbasis
def findNeelstate(length,particlenumber,table):
    """find positions of Neel states in table"""
    assert length==2*particlenumber, "need exactly half filling for Neel state"
    tot = 2**length-1
    N2 = tot//3   # integer representation of the Neel state 010101...
    N1 = tot-N2   # integer representation of the other Neel state 101010...
    return findIndex(table,N1), findIndex(table,N2)


def calc_LE(v,U,Neelpos,tint,tmax,dt): # return rate starting from Neel state
    t = np.arange(tint,tmax,dt)
    """compute time evolution from Neel state"""
    vNeel=U[Neelpos[0]].conj()     # Neel state |1010> in eigenbasis |c1 c2 ...>
    LE = np.zeros(len(t))
    evol = np.exp(-1j*t[:,np.newaxis]*v[np.newaxis])
    """ <vNeel*U|U*exp(-iEt)*U|U*Neel>"""
    LE=np.abs(np.inner(evol*vNeel[np.newaxis,:],vNeel[np.newaxis,:]))
    return LE


def sigmaZ(table):
    L=len(table)
    Neel = np.zeros((int(L),int(L/2)))
    for i in range(0,int(L)):
        for j in range(0,int(L/2)):
            if i+1==2*(j+1)-1: Neel[i,j]=1
    CM = np.dot(Neel,Neel.transpose())          # CM in Ising basis
    for i in range(1,L,2):
        CM[i][i]=-1
    return CM

############################################################################
# Part 3
# Run the program for both cases
############################################################################
tic1=time.time()
tint1=args.tint
tmid1=args.tmid1
tmid2=args.tmid2
tmid3=args.tmid3
tmid4=args.tmid4
tmid5=args.tmid5
tmid6=args.tmid6
tmid7=args.tmid7
tmax1=args.tmax1
trange=np.asarray([tint1,tmid1,tmid2,tmid3,tmid4,tmid5,tmid6,tmid7,tmax1],float)


# calculate part 2
particlenumber=args.L/2
Psi=manyPsi(particlenumber,args.L)
table = findMagnetizationStates(args.L,particlenumber)
Neelpos = findNeelstate(args.L,particlenumber,table)
MPDW = construct_MPDW(Psi,args.L,args.W)
MPH = construct_MPH(args.V,args.L,table)
vmi,Umi = np.linalg.eigh(MPH)
Umi = Umi.transpose()
sigz = sigmaZ(table)

t=np.arange(tint1,tmid1,0.1)
dat2=0#np.zeros((args.dat,len(t)))
for i in range(args.dat):
    Store1=0
    Store2=0
    Store=np.zeros((8,args.tnum))
    for samp in range(int(args.sample)):
        MPDW = construct_MPDW(Psi,args.L,args.W)
        MPHfW = MPH + MPDW - sigz
        vmf,Umf = np.linalg.eigh(MPHfW)
        for ok in range(1,len(trange)):
            tint=trange[ok-1]
            tmax=trange[ok]
            dt=(tmax-tint)/args.tnum
            Store[ok-1] += np.squeeze(calc_LE(vmf,Umf,Neelpos,tint,tmax,dt))
    dat2 += Store/args.sample

result=dat2/args.dat
LE2=np.asarray([])
t=np.asarray([])
for ok in range(1,len(trange)):
    tint=trange[ok-1]
    tmax=trange[ok]
    dt=(tmax-tint)/args.tnum
    tnow=np.arange(tint,tmax,dt)
    t=np.concatenate((t,tnow),axis=0)
    LE2=np.concatenate((LE2,result[ok-1]),axis=0)
RR2=-2*np.log(LE2)/args.L


""" Plot the results """
if args.openbc == 1:
    BC = 'OBC'
elif args.openbc == 0:
    BC = 'PBC'
tiTle1=' and '+str(BC)

plt.plot(t,LE2,'b--',label='MPH Approach')
plt.xticks([0,1e0,1e2,1e4,1e6,1e8,1e10,1e12])
#plt.xticks([1,1e2,1e4,1e6,1e8,1e10,1e12])
plt.legend(loc='upper right');
plt.ylabel('$\mathcal{L}(t)$')
plt.xlabel('t')
tiTleLE2='Loschmidt echo for XXZ with L = '+str(args.L)+', V = '+str(args.V)
tiTleL=tiTleLE2+tiTle1
plt.title(tiTleL,fontsize=10.5)
plt.show()

plt.plot(t,RR2,'b--',label='MPH Approach')
plt.legend(loc='upper right');
plt.ylabel('l(t)')
plt.xlabel('t')
tiTleRR2='Return rate for XXZ with L = '+str(args.L)+', V = '+str(args.V)
tiTleR=tiTleRR2+tiTle1
plt.title(tiTleR,fontsize=10.5)
plt.show()


fileLE2a='XXZ_'+str(BC)+'_MPH_'+str(args.L)+'LE_del='
fileLE2b=str(args.V)+'_W='+str(args.W)+'.dat'
fileLE2 =fileLE2a+fileLE2b
g=open(fileLE2,'w')
for item in LE2:
    print(item,file=g)
g.close()


tic2=time.time()
print('Total time is ',(tic2-tic1)/60,' minutes.')
process = psutil.Process(os.getpid())
print(process.memory_info().rss)
print(str(args))

