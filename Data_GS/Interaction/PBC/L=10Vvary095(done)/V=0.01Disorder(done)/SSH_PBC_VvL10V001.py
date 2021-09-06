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
V=0.01
delta=-0.95
dt=0.01
tint=0
tmid1=2
tmid2=3
tmax=20
t1=np.arange(tint,tmid1,dt)
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

"""Plot the Figure """
wb=['1e-4','1e-3','1e-2','5e-2','1e-1']
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textr='\n'.join((
        r'(A) V: 0'+' -> '+str(V),))
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm,Qfiles[0],label=r'W=0.0')
for num in range(len(Pfiles)):
    ax.plot(tm,Pfiles[num],label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'l$^\ast$(t)')
plt.ylim(-0.05,0.8)
plt.text(0.3,0.7,textr)
ax.legend(loc='upper right',fontsize='9')
plt.show()
