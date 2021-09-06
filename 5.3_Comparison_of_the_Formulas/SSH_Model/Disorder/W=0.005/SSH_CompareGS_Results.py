# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:34:20 2017

@author: LewisCYC
"""
import os             
import glob
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

delta=-0.5
dt1=0.05
dt2=0.05
tint=0
tmid=2
tmax=10
t1=np.arange(tint,tmid,dt1)
t2=np.arange(tmid,tmax+dt2,dt2)
t=np.concatenate((t1,t2), axis=0)
cwd = os.getcwd() #current working directory
item=glob.glob("*.txt")
tm=t
Pfiles=np.zeros((len(item),len(tm)))
a=0
for i in range(len(item)):
    f=open(item[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        Pfiles[a][j]=AA[j]
    a+=1


"""Plot the Figure """
#tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' and $\mu\in$ [$-W,W$]'

Wlabel=['SPH Approach','MPH Approach']
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm,np.exp(-7*Pfiles[0]),'r',label=Wlabel[0])
ax.plot(tm,np.exp(-7*Pfiles[1]),'b--',label=Wlabel[1])
plt.xlabel('t')
plt.ylabel('$\mathcal{L}(t)$')
ax.legend(loc='upper right')
plt.show()

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm,Pfiles[0],'r',label=Wlabel[0])
ax.plot(tm,Pfiles[1],'b--',label=Wlabel[1])
plt.xlabel('t')
plt.ylabel('Return rate l(t)')
ax.legend(loc='upper right')
plt.show()
