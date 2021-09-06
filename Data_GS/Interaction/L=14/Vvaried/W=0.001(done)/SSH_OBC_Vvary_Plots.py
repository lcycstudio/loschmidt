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
delta=-0.5
W=0.001
V=1.0
tint=0
tmax=50
dt=0.25
tm=np.arange(tint,tmax+dt/2,dt)

cwd = os.getcwd()
item=glob.glob("*.dat")
Pfiles=np.zeros((len(item),len(tm)))
a=0
for i in item:
    print(i)
    f = open(str(i), "r")
    lines = f.readlines()
    f.close()
    print('i: ',len(lines))
    for k in range (0,len(lines)):
        Pfiles[a][k]=float(lines[k])
    a+=1

Sent=0
for i in range(len(Pfiles)):
    Sent+=Pfiles[i]
RRF=np.divide(Sent,len(Pfiles))

f = open('SSH_OBC_Vvary_L='+str(L)+'D='+str(delta)+'W='+str(W)+'.txt','w')
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

