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
delta=0.1
tmax=100
dt=0.005
openbc=0
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
f.close()

# Plot the result
tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' --> -'+str(delta)+' and '+bc
#plt.plot(tm,Pfiles[len(item)-1],label=item[len(item)-1][15:-4])
a=0
for num in range(len(Pfiles)):
    if num==1:
        plt.plot(tm,Pfiles[num],label=item[a][15:-4]+' (fix)')
    elif num==2:
        plt.plot(tm,Pfiles[num],label='V=0.0-->'+item[a][20:-4])
    else:
        plt.plot(tm,Pfiles[num],label=item[a][15:-4])
    a+=1
plt.xlabel('Time $t$')
plt.ylabel('Return Rate $l(t)$')
plt.title(tiTle,fontsize=11)
plt.legend(loc='upper right')
plt.show()


tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' --> -'+str(delta)+' and '+bc
b=0
for num in range(len(Pfiles)):
    if num==1:
        plt.plot(tm,Pfiles[num],label=item[b][15:-4]+' (fix)')
    elif num==2:
        plt.plot(tm,Pfiles[num],label='V=0.0-->'+item[b][20:-4])
    else:
        plt.plot(tm,Pfiles[num],label=item[b][15:-4])
    b+=1
plt.xlim([0,10])
plt.xlabel('Time $t$')
plt.ylabel('Return Rate $l(t)$')
plt.title(tiTle,fontsize=11)
plt.legend(loc='upper right')
plt.show()
