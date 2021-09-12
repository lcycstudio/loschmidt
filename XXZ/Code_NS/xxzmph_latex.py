"""
Created by Ye Cheng (Lewis) Chen and Jesko Sirker
The disordered XXZ model 
Calculation for Loschmidt echo using SPH and formula |det(1-C+C*exp(-iHt))|^2
"""
import numpy as np
import argparse
from bisect import bisect_left as findIndex

# parameters for entanglement calculations
parser = argparse.ArgumentParser(description=
'Calculation of Loschmidt echo for disordered fermions')
parser.add_argument('-L', type=int, default=4,
                    help='# system length L')
parser.add_argument('-V', type=float, default=1.0,
                    help='V interaction term')
parser.add_argument('-W', type=float, default=0.0,
                    help='width box potential disorder')
parser.add_argument('-tint', type=float, default=0,
                    help='# initial time')
parser.add_argument('-tmax', type=float, default=50,
                    help='# maximum time range')
parser.add_argument('-dt', type=float, default=0.005,
                    help='# discrete time interval')
parser.add_argument('-sample', type=int, default=1,
                    help='# samples')
parser.add_argument('-openbc', type=int, default=1,
                    help='OBC = 1, PBC = 0')
args=parser.parse_args()

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
    t = np.arange(tint,tmax+dt/2,dt)
    """compute time evolution from Neel state"""
    vNeel=U[Neelpos[0]].conj()     # Neel state |1010> in eigenbasis |c1 c2 ...>
    LE = np.zeros(len(t))
    evol = np.exp(-1j*t[:,np.newaxis]*v[np.newaxis])
    """ <vNeel*U|U*exp(-iEt)*U|U*Neel>"""
    LE=np.abs(np.inner(evol*vNeel[np.newaxis,:],vNeel[np.newaxis,:]))
    return LE

# Run the program for both cases
t=np.arange(args.tint,args.tmax+args.dt/2,args.dt)

# calculate part the many-particle Hamiltonian
particlenumber=args.L/2
Psi=manyPsi(particlenumber,args.L)
table = findMagnetizationStates(args.L,particlenumber)
Neelpos = findNeelstate(args.L,particlenumber,table)
MPDW = construct_MPDW(Psi,args.L,args.W)
MPH = construct_MPH(args.V,args.L,table)
vmi,Umi = np.linalg.eigh(MPH) 
Umi = Umi.transpose()


for i in range(args.dat):
    Store2=0
    for samp in range(int(args.sample)):
        MPDW = construct_MPDW(Psi,args.L,args.W)
        MPHfW = MPH + MPDW
        vmf,Umf = np.linalg.eigh(MPHfW) 
        Store2 += calc_LE(vmf,Umi,Neelpos,args.tint,args.tmax,args.dt)

LE2=np.squeeze(Store2/args.sample)
RR2=-2*np.log(LE2)/args.L

for item in RR2:
    print(item)