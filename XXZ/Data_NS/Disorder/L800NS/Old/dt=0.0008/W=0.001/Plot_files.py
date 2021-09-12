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
delta=0.3
W=0.001
tmid=2
tmax=10
dt1=0.0008
dt2=0.05
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
Sent=0
for i in range(len(Pfiles)):
    Sent+=Pfiles[i]
LEF=np.divide(Sent,len(Pfiles))
RRF=-2*np.log(LEF)/L

f = open('MPH_L='+str(L)+'D='+str(delta)+'W='+str(W)+'.txt','w')
for step in range (0,len(tm)):
    print(RRF[step],file=f)
f.close()

"""Plot the Figure """
tiTle='Return Rate for L = 800 with $\delta$ = '+str(delta)+' and W = '+str(W)
plt.plot(tm,RRF)#label=item[a][16:-4])
#plt.xlim([0,14])
plt.xlabel('Time $t$')
plt.ylabel('Return Rate $l(t)$')
plt.title(tiTle,fontsize=11)
plt.legend(loc='upper right')

