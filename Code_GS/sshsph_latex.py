"""
Created by Ye Cheng (Lewis) Chen and Jesko Sirker
The non-interacting disordered SSH model 
Calculation for Loschmidt echo using SPH and formula |det(1-C+C*exp(-iHt))|^2
"""
import numpy as np
import argparse

# parameters for entanglement calculations
parser = argparse.ArgumentParser(description=
'Calculation of Loschmidt echo for disordered fermions')
parser.add_argument('-L', type=int, default=4,
                    help='# system length L')
parser.add_argument('-W', type=float, default=0.0,
                    help='width box potential disorder')
parser.add_argument('-delta', type=float, default=-0.3,
                    help='# quench parameter delta')
parser.add_argument('-dt', type=float, default=0.01,
                    help='# discrete time interval first part')
parser.add_argument('-tint', type=float, default=0,
                    help='# maximum time range first part')
parser.add_argument('-tmax', type=float, default=20,
                    help='# maximum time range second part')
parser.add_argument('-sample', type=int, default=1,
                    help='# samples')
parser.add_argument('-openbc', type=int, default=1,
                    help='OBC = 1, PBC = 0')
args=parser.parse_args()

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
    Ut = np.dot(U.conj(),np.dot(np.diag(np.exp(-1j*v*t)),(U.transpose()).conj()))
    return Ut

# construct two point correlation matrix
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

# Run the program for both cases
t=np.arange(args.tint,args.tmax+args.dt/2,args.dt)

# calculate single-particle Hamiltonian
SPHi = construct_SPH(args.delta,args.L,args.openbc)
SPHf = construct_SPH(-args.delta,args.L,args.openbc)

Store1=0
for samp in range(int(args.sample)):
    APDW = construct_APDW(args.L,args.W)
    SPHiW = SPHi + APDW
    SPHfW = SPHf + APDW
    vsi,Usi = np.linalg.eigh(SPHiW)
    vsf,Usf = np.linalg.eigh(SPHfW)
    CM = construct_CM(Usi,args.L)
    Store1 += calc_detLE(vsf,Usf,CM,t)

LE1=np.squeeze(Store1/args.sample)
RR1=-2*np.log(LE1)/args.L # return rate

for item in RR1:
    print(item)
    