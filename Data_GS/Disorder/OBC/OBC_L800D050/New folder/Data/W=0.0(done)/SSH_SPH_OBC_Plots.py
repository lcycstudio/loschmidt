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
W=0.0
tmid=2
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
Sent=0
for i in range(len(Pfiles)):
    Sent+=Pfiles[i]
LEF=np.divide(Sent,len(Pfiles))
RRF=-2*np.log(LEF)/L


""" Smoothing out W=0.0 """
maxIndex=RRF.tolist().index(np.amax(RRF))
tn=np.arange(maxIndex,2000,30)
tIndex=np.concatenate((tn,np.array([2000])), axis=0)
x=np.zeros(len(tIndex))
y=np.zeros(len(x))
for jj in range(len(x)):
    y[jj]=RRF[tIndex[jj]]
    x[jj]=tm[tIndex[jj]]
z=np.polyfit(x,y,2)
p=np.poly1d(z)
dt3=len(RRF[maxIndex:2000])
xp=np.linspace(x[0],x[-1],dt3)
yp=p(xp)
fig = plt.figure()
ax = plt.subplot(111)
Wlabel=['0.0','5e-7','5e-6','5e-5']
ax.plot(xp,p(xp),'k:',label='Polyfit')
RRF[maxIndex:2000]=yp
tm0=tm
tm[maxIndex:2000]=xp


f = open('SSH_SPH_L='+str(L)+'D='+str(delta)+'W='+str(W)+'.txt','w')
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

