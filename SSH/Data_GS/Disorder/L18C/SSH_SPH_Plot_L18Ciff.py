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

L=18
delta=-0.95
dt1=0.1
dt2=1e-5
dt3=0.01
tint=1.6
tmid1=1.61
tmid2=1.7
tmax=1.71
t1=np.arange(tint,tmid1,dt1)
t2=np.arange(tmid1,tmid2,dt2)
t3=np.arange(tmid2,tmax+dt3,dt3)
tt=np.concatenate((t1,t2), axis=0)
tm=np.concatenate((tt,t3), axis=0)
cwd = os.getcwd() #current working directory
item=glob.glob("*.dat")
item2=glob.glob("*.txt")
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

dy = np.zeros(Pfiles.shape,np.float)
dy[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])

dy2 = np.zeros(Qfiles.shape,np.float)
dy2[:,0:-1] = np.diff(Qfiles)/np.diff(tm)
dy2[:,-1] = (Qfiles[:,-1] - Qfiles[:,-2])/(tm[-1] - tm[-2])

"""Plots of the 1st Derivative"""
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textr='\n'.join((
        r'$\delta$: '+str(delta)+' -> +'+str(-delta),
        r'L = '+str(L)))
plt.plot(tm,dy[0],label=item[0][25:-9])
for num in range(len(Pfiles)-1,0,-1):
    plt.plot(tm,dy[num],':',label=item[num][25:-9])
plt.xlabel('t')
plt.ylabel(r'dl$^\ast$/dt')
plt.text(1.66,10,textr,bbox=props)
plt.legend(loc='upper right')#bbox_to_anchor=(1.0, 1.02),ncol=1)
plt.show()


