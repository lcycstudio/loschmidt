# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:34:20 2017

@author: LewisCYC
"""
import os             
import glob
import numpy as np
import matplotlib.pyplot as plt

L=800
delta=-0.5
W=0.005
tmid=3
tmax=10
dt1=0.001
dt2=0.1
cwd = os.getcwd() #current working directory
item=glob.glob("*.dat")
t1=np.arange(0,tmid,dt1)
t2=np.arange(tmid,tmax+dt2,dt2)
tm=np.concatenate((t1,t2), axis=0)
Pfiles=np.zeros((len(item),len(tm)))
a=0
for i in item:
    print(i)
    f = open(str(i), "r")
    lines = f.readlines()
    print('i: ',len(lines))
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
Store=0
for i in range(int(size)):
    Store+=newdat[i]


RRF=np.divide(Store,size)


f = open('XXZ_SPH_L='+str(L)+'D='+str(delta)+'W='+str(W)+'.txt','w')
for step in range (0,len(tm)):
    print(RRF[step],file=f)
f.close()

"""Plot the Figure """
tiTle='Return Rate for the XXZ model with L = 800 and W = '+str(W)
plt.plot(tm,RRF)
plt.xlabel('Time $t$')
plt.ylabel('Return Rate $l(t)$')
plt.title(tiTle,fontsize=11)
plt.legend(loc='upper right')

