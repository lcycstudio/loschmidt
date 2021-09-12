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
#W=1.0
tint=0.7854
tmid=0.7856
tmax=0.7858
dt1=2e-7
dt2=2e-7
cwd = os.getcwd() #current working directory
item=glob.glob("*.txt")
t1=np.arange(tint,tmid,dt1)
t2=np.arange(tmid,tmax+dt2,dt2)
tm=np.concatenate((t1,t2), axis=0)
t1=np.arange(0.7852,tmid,dt1)
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
c=['b','y','r','g']
plt.plot(tm[150:350],Pfiles[3][150:350],label=item[3][14:-4],color=c[0])
Wlabel=['0.00005','0.00025','0.0005']
a=0
for num in range(3):
    #c1=np.random.rand()
    #c2=np.random.rand()
    #c3=np.random.rand()
    plt.plot(tm[150:350],Pfiles[num][150:350],label='W='+Wlabel[num],color=c[a+1])#color=(c1,c2,c3))
    a+=1

plt.ylim([1.594,1.625])
plt.xlabel('t')
plt.ylabel('l(t)')
plt.legend(loc='upper right')
plt.xticks([0.78543,0.78545,0.78547])
plt.show()


tiTle='Return Rate for L = '+str(L)+' with $\delta$ = '+str(delta)+' and W $\in$ [$W/2,W/2$]'
b=0
tn=np.arange(0.7852,tmax+dt2,dt2)
d=open('SPH_L=800D=0.3W=0.0.dat','r')
BB=d.readlines()
RR=np.zeros(len(BB))
for k in range(len(BB)):
    RR[k]=BB[k]
    
    
    
plt.plot(tn,RR,label=item[3][14:-4],color=c[0])
for num in range(3):
    plt.plot(tm,Pfiles[num],label='W='+Wlabel[num],color=c[b+1])#color=(c1,c2,c3))
    b+=1

plt.xlim([0.7854,0.7855])
plt.ylim([1.595,1.625])
plt.xlabel('t')
plt.ylabel('l(t)')
plt.legend(loc='upper right')
plt.show()

b=0
plt.plot(tn,RR,label=item[3][14:-4],color=c[0])
for num in range(3):
    plt.plot(tm,Pfiles[num],label='W='+Wlabel[num],color=c[b+1])#color=(c1,c2,c3))
    b+=1

plt.xlabel('t')
plt.ylabel('l(t)')
plt.legend(loc='lower right')
plt.show()
