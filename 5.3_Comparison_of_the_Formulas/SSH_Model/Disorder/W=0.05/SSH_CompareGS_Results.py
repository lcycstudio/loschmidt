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
dt1=0.1
dt2=0.1
tint=0
tmid=10
tmax=20
t1=np.arange(tint,tmid,dt1)
t2=np.arange(tmid,tmax+dt2,dt2)
t=np.concatenate((t1,t2), axis=0)
cwd = os.getcwd() #current working directory
item=glob.glob("*.dat")
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

Wlabel=['SPHA','MPHA']
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm,Pfiles[0],'r',label=Wlabel[0])
ax.plot(tm,Pfiles[1],'b--',label=Wlabel[1])
plt.xlabel('Time $t$')
plt.ylabel('l(t)')
ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1)
plt.show()

LE1=np.exp(-14*Pfiles[0]/2)
LE2=np.exp(-14*Pfiles[1]/2)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm,LE1,'r',label=Wlabel[0])
ax.plot(tm,LE2,'b--',label=Wlabel[1])
plt.xlabel('Time $t$')
plt.ylabel(r'$\mathcal{L}(t)$')
ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1)
plt.show()