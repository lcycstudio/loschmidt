# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:34:20 2017

@author: LewisCYC
"""
import os             
import glob
import numpy as np
import matplotlib.pyplot as plt

L=14
dt=0.005
delta=0.0
W=0.5
cwd = os.getcwd() #current working directory
item=glob.glob("*.dat")
tm=np.arange(0,10,dt)
Pfiles=np.zeros((len(item),len(tm)))
a=0
for i in item:
    print(i)
    f = open(str(i), "r")
    lines = f.readlines()
    for k in range (0,len(tm)):
        Pfiles[a][k]=float(lines[k])
    f.close()
    a+=1

size=np.divide(len(Pfiles)*(len(Pfiles)-1),2.0)
newdat=np.zeros((int(size),len(tm)))
a=0
b=0
for i in Pfiles:
    for j in range(a+1,len(Pfiles)):
        newdat[b]=np.divide(i+Pfiles[j],2.0)
        b=b+1
    a=a+1

Sent=0
for i in range(int(size)):
    Sent+=newdat[i]
LEF=np.divide(Sent,size)
RRF=-np.log(LEF)/L

f = open('MPH_L='+str(L)+'W='+str(W)+'D='+str(delta)+'.txt','w')
for step in range (0,len(tm)):
    print(RRF[step],file=f)
f.close()

"""Plot the Figure """
item=glob.glob("*.txt")
Qfiles=np.zeros((len(tm)))
f = open(item[1], "r")
lines = f.readlines()
for k in range (0,len(tm)):
    Qfiles[k]=float(lines[k])
f.close()


plt.plot(tm,Qfiles,'r',label='SPH Approach')
plt.plot(tm,RRF,'b--',label='MPH Approach')
plt.legend(loc='upper right');
plt.ylabel('l(t)')
plt.xlabel('t')
plt.ylim((-0.05,1.4))
#plt.title('Return Rate for L = '+str(L)+' with $\delta$ = '+str(delta)+' and W = '+str(W),fontsize=12)
