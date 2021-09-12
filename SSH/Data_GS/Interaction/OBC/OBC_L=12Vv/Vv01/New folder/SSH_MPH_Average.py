# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:34:20 2017

@author: LewisCYC
"""
import os             
import glob
import numpy as np
import matplotlib.pyplot as plt

L=12
V=0.1   
delta=0.95
W=5.0
openbc=1
Vfix=0
tint=0
tmax=20
dt1=0.005
cwd = os.getcwd() #current working directory
item=glob.glob("*.dat")
tm=np.arange(tint,tmax+dt1/2,dt1)
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

if openbc==1:
    bc = '_OBC'
elif openbc==0:
    bc = '_PBC'
if Vfix==1:
    vf = '_VFix='
elif Vfix== 0:
    vf = '_VVary='

f = open('SSH'+bc+vf+str(V)+'_MPH_'+str(L)+'LE_del='+str(delta)+'_W='+str(W)+'.txt','w')
for step in range (0,len(tm)):
    print(RRF[step],file=f)
f.close()

"""Plot the Figure """
tiTle='Return Rate for L = 800 with $\delta$ = '+str(delta)+' and W = '+str(W)
plt.plot(tm,RRF)
plt.xlabel('Time $t$')
plt.ylabel('Return Rate $l(t)$')
plt.title(tiTle,fontsize=11)
plt.legend(loc='upper right')

