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

# parameters for entanglement calculations
parser = argparse.ArgumentParser(description=
'Calculation of Loschmidt echo for disordered fermions')
parser.add_argument('-L', type=int, default=800,
                    help='# system length L')
parser.add_argument('-V', type=float, default=0.0,
                    help='V interaction term')
parser.add_argument('-W', type=float, default=0.0,
                    help='width box potential disorder')
parser.add_argument('-delta', type=float, default=0.95,
                    help='# quench parameter delta')
parser.add_argument('-dt1', type=float, default=0.01,
                    help='# discrete time interval first part')
parser.add_argument('-dt2', type=float, default=0.01,
                    help='# discrete time interval second part')
parser.add_argument('-tint', type=float, default=0,
                    help='# maximum time range first part')
parser.add_argument('-tmid', type=float, default=2,
                    help='# maximum time range first part')
parser.add_argument('-tmax', type=float, default=10,
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
    if delta == 1:
        for i in range(0,L-1,2):
            H[i,i+1]=1.0
            H[i+1,i]=1.0
    elif delta == -1:
        for i in range(1,L-1,2):
            H[i,i+1]=1.0
            H[i+1,i]=1.0
            if openbc == 0:
                H[0][-1]=1.0#+delta*(-1)**L
                H[-1][0]=1.0#+delta*(-1)**L
    else:
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
def calc_M(v,U,CM,t):
    M=np.zeros(CM.shape)
    eigvM=np.zeros((len(t),args.L),dtype=complex)
    for i in t:
        Ut = construct_U(v,U,i)
        k=t.tolist().index(i)
        M=np.identity(args.L)-CM+np.dot(CM,Ut)
        eigvM[k]=np.linalg.eig(M)[0]
    return eigvM


def calc_detLE(v,U,CM,t):
    LE=np.zeros(len(t))
    M=np.zeros(CM.shape)
    for i in t:
        Ut = construct_U(v,U,i)
        k=t.tolist().index(i)
        M=np.identity(args.L)-CM+np.dot(CM,Ut)
        LE[k]=np.abs(np.linalg.det(M))
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
SPHi = construct_SPH(args.delta,args.L,args.openbc)
SPHf = construct_SPH(-args.delta,args.L,args.openbc)


dat1=np.zeros((args.dat,len(t),args.L),dtype=complex)
dat2=np.zeros((args.dat,len(t)))
for i in range(args.dat):
    Store1=0
    Store2=0
    for samp in range(int(args.sample)):
        APDW = construct_APDW(args.L,args.W)
        SPHiW = SPHi + APDW
        SPHfW = SPHf + APDW
        vsi,Usi = np.linalg.eigh(SPHiW)
        vsf,Usf = np.linalg.eigh(SPHfW)
        CM = construct_CM(Usi,args.L)
        Store1 += calc_M(vsf,Usf,CM,t)
        Store2 += calc_detLE(vsf,Usf,CM,t)
    dat1[i] += np.squeeze(Store1/args.sample)
    dat2[i] += np.squeeze(Store2/args.sample)

size=np.divide(args.dat*(args.dat-1),2.0)
newdat1=np.zeros((int(size),len(t),args.L))
a=0
b=0
# Averaging LE2
for k in dat1:
    for j in range(a+1,len(dat1)):
        newdat1[b]=np.divide(k+dat1[j],2.0)
        b=b+1
    a=a+1
eigvM=0
for k in range(0,int(size)):
    eigvM+=newdat1[k]
if args.dat==1:
    result1=dat1[0]
    result2=dat2[0]
elif args.dat > 1:
    result1=np.divide(eigvM,size)

if args.openbc == 1:
    BC = 'OBC'
    color ='b:'
elif args.openbc == 0:
    BC = 'PBC'
    color = 'g:'
    
    
eigvR=result1.real
eigvI=result1.imag
#plt.plot(t,np.abs(eigvM.transpose()[0]),'g:',label=BC+'(W='+str(args.W)+')')
plt.plot(t,np.abs(eigvR.transpose()[0]),':',label=BC+'(W='+str(args.W)+')')#+str(args.W)+')')
plt.plot(t,np.abs(eigvR),color)
plt.legend(loc='upper right')
plt.ylabel(r'$|\lambda_j|/L$')
plt.xlabel('t')
plt.show()
ee=np.zeros(len(t))
for i in range(len(ee)):
    ee[i]=np.sum(eigvR[i])/args.L
LE1=result2
RR1=-2*np.log(LE1)/args.L
plt.plot(t,ee,':',label=r'$|\lambda_j|/L$');
plt.plot(t,RR1,label=r'$\mathcal{L}(t)$');
plt.plot(t,LE1,label='$l(t)$');
plt.legend(loc='upper right')
plt.show()


a=0
for i in range(len(RR1)):
    if np.abs(RR1[i]-ee[i])<0.011:
        print(i)
        a=a+1
ttc=np.zeros(a)
a=0
for i in range(len(RR1)):
    if np.abs(RR1[i]-ee[i])<0.011:
        ttc[a]=t[i]
        a=a+1
#for k in range(len(ttc)):
 #   ttc[k]=t[ttc[k]]

LE1=result2
RR1=-2*np.log(LE1)/args.L
#plt.plot(t,ee,':',label=r'$|\lambda_j|/L$');
plt.plot(t,RR1,label=r'$\mathcal{L}(t)$');
#plt.plot(t,LE1,label='$l(t)$');
plt.legend(loc='upper right')
for l in range(len(ttc)):
    plt.axvline(ttc[l],linestyle=':')
plt.show()


#plt.plot(t,np.abs(eigvR.transpose()[0]),':',label=BC+'(W='+str(args.W)+')')#+str(args.W)+')')
#plt.plot(t,np.abs(eigvR),color)
#plt.legend(loc='upper right')
#plt.ylabel(r'$|\lambda_j|/L$')
#plt.xlabel('t')
#for l in range(len(ttc)):
#    plt.axvline(ttc[l],linestyle=':')
#plt.show()


fileRR1a='SSH_'+str(BC)+'_SPH_'+str(args.L)+'RR_del='
fileRR1b=str(args.delta)+'_W='+str(args.W)+'_L='+str(args.L)+'.dat'
fileRR1 =fileRR1a+fileRR1b
f=open(fileRR1,'w')
for item in RR1:
    print(item,file=f)
f.close()


fileEV1a='SSH_'+str(BC)+'_SPH_'+str(args.L)+'EV_del='
fileEV1b=str(args.delta)+'_W='+str(args.W)+'_L='+str(args.L)+'.dat'
fileEV1 =fileEV1a+fileEV1b
fileEV1 =fileEV1a+fileEV1b
f=open(fileEV1,'w')
for item in ee:
    print(item,file=f)
f.close()


tic2=time.time()
print('Total time is ',(tic2-tic1)/60,' minutes.')
process = psutil.Process(os.getpid())
print(process.memory_info().rss)
print(str(args))
