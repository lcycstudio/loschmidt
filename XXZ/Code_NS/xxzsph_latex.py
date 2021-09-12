"""
Created by Ye Cheng (Lewis) Chen and Jesko Sirker
The disordered XX model (non-interacting)
Calculation for Loschmidt echo using SPH and formula |det(1-C+C*exp(-iHt))|^2
"""
import numpy as np
import argparse

# parameters for entanglement calculations
parser = argparse.ArgumentParser(description=
'Calculation of Loschmidt echo for disordered fermions')
parser.add_argument('-L', type=int, default=4,
                    help='# system length L')
parser.add_argument('-W', type=float, default=1.0,
                    help='width box potential disorder')
parser.add_argument('-dt', type=float, default=0.01,
                    help='# discrete time interval first part')
parser.add_argument('-tint', type=float, default=0,
                    help='# maximum time range first part')
parser.add_argument('-tmax', type=float, default=20,
                    help='# maximum time range second part')
parser.add_argument('-sample', type=int, default=30,
                    help='# samples')
parser.add_argument('-openbc', type=int, default=1,
                    help='OBC = 1, PBC = 0')
args=parser.parse_args()

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
def construct_SPH(L,openbc):
    H = np.zeros((L,L))
    for i in range(0,L-1):
        H[i,i+1]=0.5
        H[i+1,i]=0.5
    if openbc == 0:
        H[0][-1]=0.5
        H[-1][0]=0.5
    return H

# construct unitary time evolution operator Uexp(-iDt)U*
def construct_U(v,U,t):
    Ut = np.dot(U.conj(),np.dot(np.diag(np.exp(-1j*v*t)),(U.transpose()).conj()))
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


# Run the program 
t=np.arange(args.tint,args.tmax+args.dt/2,args.dt)

# calculate part the single-particle Hamiltonian
SPH = construct_SPH(args.L,args.openbc)
vs,Us = np.linalg.eigh(SPH)
CM = construct_CM(args.L)

Store1=0
for samp in range(int(args.sample)):
    APDW = construct_APDW(args.L,args.W)
    SPHfW = SPH + APDW
    vsf,Usf = np.linalg.eigh(SPHfW)
    Store1 += calc_detLE(vsf,Usf,CM,t)
    
LE1=np.squeeze(Store1/args.sample)
RR1=-2*np.log(LE1)/args.L

for item in RR1:
    print(item)

