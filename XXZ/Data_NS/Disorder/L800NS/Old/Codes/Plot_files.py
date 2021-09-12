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
W=1.0
tmid=2
tmax=10
dt1=0.0008
dt2=0.05
cwd = os.getcwd() #current working directory
item=glob.glob("*.txt")
t1=np.arange(0,tmid,dt1)
t2=np.arange(tmid,tmax+dt2,dt2)
tm=np.concatenate((t1,t2), axis=0)
Pfiles=np.zeros((len(item),len(tm)))
a=0
for i in range(len(item)):
    f=open(item[i],'r')
    AA=f.readlines()
    for j in range(len(AA)):
        Pfiles[a][j]=AA[j]
    a+=1


"""Plot the Figure """
tiTle='Return Rate for L = '+str(L)+' with $\delta$ = '+str(delta)+' and W $\in$ [$W/2,W/2$]'
a=0
for result in Pfiles:
    #c1=np.random.rand()
    #c2=np.random.rand()
    #c3=np.random.rand()
    plt.plot(tm,result,label=item[a][14:-4])#color=(c1,c2,c3))
    a+=1

plt.xlim([0,15])
plt.xlabel('Time $t$')
plt.ylabel('Return Rate $l(t)$')
plt.title(tiTle,fontsize=11)
plt.legend(loc='upper right')
plt.show()


tiTle='Return Rate for L = '+str(L)+' with $\delta$ = '+str(delta)+' and W $\in$ [$W/2,W/2$]'
b=0
for result in Pfiles:
    plt.plot(tm[981:984],result[981:984],label=item[b][14:-4])#color=(c1,c2,c3))
    b+=1
plt.xlim([0.7852,0.7864])
plt.xlabel('Time $t$')
plt.ylabel('Return Rate $l(t)$')
plt.title(tiTle,fontsize=11)
plt.legend(loc='upper right')
plt.show()
