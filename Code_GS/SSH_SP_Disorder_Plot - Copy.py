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
parser.add_argument('-L', type=int, default=60,
                    help='# system length L')
parser.add_argument('-V', type=float, default=0,
                    help='V interaction term')
parser.add_argument('-W', type=float, default=0.0,
                    help='width box potential disorder')
parser.add_argument('-delta', type=float, default=-0.5,
                    help='# quench parameter delta')
parser.add_argument('-dt1', type=float, default=0.01,
                    help='# discrete time interval first part')
parser.add_argument('-dt2', type=float, default=0.01,
                    help='# discrete time interval second part')
parser.add_argument('-dt3', type=float, default=0.01,
                    help='# discrete time interval second part')
parser.add_argument('-tint', type=float, default=0,
                    help='# maximum time range first part')
parser.add_argument('-tmid1', type=float, default=2,
                    help='# maximum time range first part')
parser.add_argument('-tmid2', type=float, default=5,
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
def calc_detLE(v,U,CM,t):
    LE=np.zeros(len(t))
    for i in t:
        Ut = construct_U(v,U,i)
        k=t.tolist().index(i)
        LE[k]=np.abs(np.linalg.det(np.identity(args.L)-CM+np.dot(CM,Ut)))
    return LE


############################################################################
# Part 3
# Calculation for Loschmidt echo using analytical equation
############################################################################
def DkList(M,delta):
    DkList = np.zeros((int(M/2),3))
    for j in range(1,int(M/2+1)):
        Dk=np.array([[-2*np.cos(2*np.pi*j/M),2*delta*np.sin(2*np.pi*j/M),0]])
        normDk=Dk/np.linalg.norm(Dk)
        DkList[j-1]=normDk
    return DkList

def EkList(M,delta):
    EkList = np.zeros(int(M/2))
    for j in range(1,int(M/2+1)):
        EkList[j-1]=2*np.sqrt(np.cos(2*np.pi*j/M)**2+delta**2*np.sin(2*np.pi*j/M)**2)
    return EkList



def calc_ASLE(t,M,delta):
    Dk1=DkList(M,delta)
    Dk2=DkList(M,-delta)
    ek=EkList(M,delta)
    #LE = np.cos(ek[0]*t)+np.dot(1j*Dk1[0],Dk2[0])*np.sin(ek[0]*t)
    LE = np.zeros(len(t))
    for i in t:
        ii = t.tolist().index(i)
        number = 1.0
        for k in range(int(M/2)):            
            number = np.multiply(number,np.abs(np.cos(ek[k]*i)+np.dot(1j*Dk1[k],Dk2[k])*np.sin(ek[k]*i)))
        LE[ii] = number
    return np.abs(LE)


############################################################################
# Part 3
# Run the program for both cases
############################################################################
tic1=time.time()
t1=np.arange(args.tint,args.tmid1,args.dt1)
t2=np.arange(args.tmid1,args.tmid2,args.dt2)
t3=np.arange(args.tmid2,args.tmax+args.dt3,args.dt3)
tt=np.concatenate((t1,t2), axis=0)
t=np.concatenate((tt,t3), axis=0)
# calculate part 1
SPHi = construct_SPH(args.delta,args.L,args.openbc)
SPHf = construct_SPH(-args.delta,args.L,args.openbc)

if args.openbc==0:
    if args.W==0:
        aLE = calc_ASLE(t,args.L,args.delta)
        aRR = -2*np.log(aLE)/args.L

    
dat1=np.zeros((args.dat,len(t)))
Store2=0
APDW=0
for i in range(args.dat):
    for samp in range(int(args.sample)):
        APDW += construct_APDW(args.L,args.W)
APDW = APDW/(args.dat*args.sample)
SPHiW = SPHi + APDW
SPHfW = SPHf + APDW


vsi,Usi = np.linalg.eigh(SPHiW)
vsf,Usf = np.linalg.eigh(SPHfW)
CM = construct_CM(Usi,args.L)
LE1= calc_detLE(vsf,Usf,CM,t)
Store2 = np.abs(vsf[int(args.L/2)])
#dat1[i] += np.squeeze(Store1/args.sample)

#LE1=result1
RR1=-2*np.log(LE1)/args.L

if args.openbc == 1:
    openbc = 'OBC'
elif args.openbc == 0:
    openbc = 'PBC'

""" Plot the results """
if args.openbc == 1:
    BC = 'OBC'
elif args.openbc == 0:
    BC = 'PBC'
tiTle1='-->-'+str(args.delta)+' and '+str(BC)
if args.delta < 0:
    tiTle1='-->+'+str(np.abs(args.delta))+' and '+str(BC)
plt.plot(t,LE1,'r',label='SPH Approach')
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
if args.openbc==0.:
    if args.W==0.:
        plt.plot(t,aRR,':',label='Analytic')
#plt.legend(loc='upper right');
plt.ylabel('l(t)')
plt.xlabel('t')
if args.delta < 0:
    tiTleRR2='Return rate for SSH with L = '+str(args.L)+', $\delta$ = '+str(args.delta)
elif args.delta > 0:
    tiTleRR2='Return rate for SSH with L = '+str(args.L)+', $\delta$ = +'+str(args.delta)
tiTleR=tiTleRR2+tiTle1
#plt.title(tiTleR,fontsize=10.5)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textr='\n'.join((
        r'$\delta$: '+str(args.delta)+' -> +'+str(-args.delta),
        r'L = '+str(args.L)))
#plt.text(6,3,textr,bbox=props)
plt.show()

fileRR1a='SSH_'+str(BC)+'_SPH_RR_del='
fileRR1b=str(args.delta)+'_W='+str(args.W)+'_L='+str(args.L)+'.dat'
fileRR1 =fileRR1a+fileRR1b
f=open(fileRR1,'w')
for item in RR1:
    print(item,file=f)
f.close()


tc1=0
tc1=np.amax(RR1)
for j in range(len(RR1)):
    if tc1==RR1[j]:
        tc1=t[int(j)]
print('tc: ',tc1)
tic2=time.time()
print('Total time is ',(tic2-tic1)/60,' minutes.')
process = psutil.Process(os.getpid())
print(process.memory_info().rss)
print(str(args))

print(vsf[int(args.L/2)])
print(Store2/args.dat/args.sample)

for i in range(len(RR1)):
    if RR1[i]==np.amax(RR1):
        print(i)
