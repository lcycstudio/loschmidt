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
V=1.0
W=6.0
tint=0
tmax=50
dt=0.005    
openbc=0
if openbc==1:
    bc='Open BC'
else:
    bc='PBC'
cwd = os.getcwd() #current working directory
item=glob.glob("*.dat")
tm=np.arange(0,tmax+dt/2,dt)
Pfiles=np.zeros((len(item),len(tm)))
a=0
for i in item:
    print(i)
    f = open(str(i), "r")
    lines = f.readlines()
    f.close()
    print('i: ',len(lines))
    for k in range (0,len(tm)):
        Pfiles[a][k]=float(lines[k])
    a+=1
Sent=0
for i in range(len(Pfiles)):
    Sent+=Pfiles[i]
RRF=np.divide(Sent,len(Pfiles))
#RRF=-2*np.log(LEF)/L

f = open('XXZ_OBC_V='+str(V)+'W='+str(W)+'.txt','w')
for step in range (0,len(tm)):
    print(RRF[step],file=f)
f.close()

"""Plot the Figure """
tiTle='Return Rate for XXZ Model with W = '+str(W)
plt.plot(tm,RRF)
plt.xlabel('Time $t$')
plt.ylabel('Return Rate $l(t)$')
plt.title(tiTle,fontsize=11)
plt.legend(loc='upper right')

