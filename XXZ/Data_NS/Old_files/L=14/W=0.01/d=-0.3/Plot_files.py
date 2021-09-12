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
delta=-0.3
W=0.01
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
LE=np.zeros((len(tm)))
f = open(item[1], "r")
lines = f.readlines()
for k in range (0,len(tm)):
    LE[k]=float(lines[k])
f.close()

RR=-2*np.log(LE)/L
tiTle='Return Rate for L = 14 with $\delta$ = '+str(delta)+' and W = '+str(W)
plt.plot(tm,RR,'r',label='SPH Approach')
plt.plot(tm,RRF,'b--',label='MPH Approach')
plt.legend(loc='upper right');
plt.ylabel('l(t)')
plt.xlabel('t')
#plt.ylim((-0.05,2.0))
#plt.title('Return Rate for L = '+str(L)+' with $\delta$ = '+str(delta)+' and W = '+str(W),fontsize=12)
