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
W=0.00
tmid=3
tmax=10
dt1=0.001
dt2=0.1
openbc=1
if openbc == 1:
    BC = 'OBC'
elif openbc == 0:
    BC = 'PBC'
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
    f.close()
    print('Data entries: ',len(lines))
    for k in range (0,len(tm)):
        Pfiles[a][k]=float(lines[k])
    a+=1
Sent=0
for i in range(len(Pfiles)):
    Sent+=Pfiles[i]
LEF=np.divide(Sent,len(Pfiles))
RRF=-2*np.log(LEF)/L

f = open('XXZ_OBC_SPH_L='+str(L)+'W='+str(W)+'.txt','w')
for step in range (0,len(tm)):
    print(RRF[step],file=f)
f.close()

"""Plot the Figure """
tiTle='Return Rate for XXZ model with W = '+str(W)+' and '+BC
plt.plot(tm,RRF)
plt.xlabel('Time $t$')
plt.ylabel('Return Rate $l(t)$')
plt.title(tiTle,fontsize=11)
plt.legend(loc='upper right')

