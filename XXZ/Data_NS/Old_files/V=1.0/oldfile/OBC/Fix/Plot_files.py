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
tmax=100
dt=0.005
openbc=1
if openbc==1:
    bc='Open BC'
else:
    bc='PBC'
cwd = os.getcwd() #current working directory
item=glob.glob("*.dat")
tm=np.arange(0,tmax,dt)
#tm=np.concatenate((t1,t2), axis=0)
Pfiles=np.zeros((len(item),len(tm)))
a=0
for i in range(len(item)):
    f=open(item[i],'r')
    AA=f.readlines()
    for j in range(len(AA)):
        Pfiles[a][j]=AA[j]
    a+=1


"""Plot the Figure """
tiTle='Return Rate for L = '+str(L)+' with V = '+str(V)+' and '+bc
#plt.plot(tm,Pfiles[len(item)-1],label=item[len(item)-1][15:-4])
a=0
for num in range(len(Pfiles)):
    plt.plot(tm,Pfiles[num],label='$\delta$=+'+item[a][11:-10]+'->-'+item[a][11:-10])
    a+=1
plt.xlabel('Time $t$')
plt.ylabel('Return Rate $l(t)$')
plt.title(tiTle,fontsize=11)
plt.legend(loc='upper right')
plt.show()


tiTle='Return Rate for L = '+str(L)+' with V = '+str(V)+' and '+bc
b=0
for num in range(len(Pfiles)):
    plt.plot(tm,Pfiles[num],label='$\delta$=+'+item[b][11:-10]+'->-'+item[b][11:-10])
    b+=1
plt.xlim([0,10])
plt.xlabel('Time $t$')
plt.ylabel('Return Rate $l(t)$')
plt.title(tiTle,fontsize=11)
plt.legend(loc='upper right')
plt.show()
