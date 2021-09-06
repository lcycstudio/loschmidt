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

L=10
delta=-0.95
dt=0.01
tint=0
tmax=20
t=np.arange(tint,tmax+dt/2,dt)
#dt2=0.05
cwd = os.getcwd() #current working directory
item=glob.glob("*.dat")
item2=glob.glob("*.txt")
tm=t
Pfiles=np.zeros((len(item),len(tm)))
Qfiles=np.zeros((len(item2),len(tm)))
a=0
for i in range(len(item)):
    f=open(item[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        Pfiles[a][j]=AA[j]
    a+=1
a=0
for i in range(len(item2)):
    f=open(item2[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        Qfiles[a][j]=AA[j]
    a+=1

Pfiles=-2*np.log(Pfiles)/L
"""Plot the Figure """
#tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' and $\mu\in$ [$-W,W$]'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textr='\n'.join((
        r'$\delta$: -0.95'+' -> '+r'+0.95',
        r'L = '+str(L)))
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm,-2*np.log(Qfiles[0])/L,label=r'V=0.0')
for num in range(len(Pfiles)):
    ax.plot(tm,Pfiles[num],label=r'V='+item[num][13:17])


plt.xlabel('t')
plt.ylabel('l(t)')
#plt.ylim(-0.05,1.0)
plt.text(7.5,3.0,textr,bbox=props)
ax.legend(bbox_to_anchor=(1.0, 1.02),ncol=1)
plt.show()
