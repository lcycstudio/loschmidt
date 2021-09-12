# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:34:20 2017

@author: LewisCYC
"""
import os             
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

delta=-0.5
dt1=0.001
dt2=0.1
tint=0
tmid=2
tmax=10
t1=np.arange(tint,tmid,dt1)
t2=np.arange(tmid,tmax+dt2,dt2)
t=np.concatenate((t1,t2), axis=0)   
#dt2=0.05
cwd = os.getcwd() #current working directory
item=glob.glob("*.dat")
tm=t
Pfiles=np.zeros((len(item),len(tm)))
a=0
for i in range(len(item)):
    f=open(item[i],'r')
    AA=f.readlines()
    for j in range(len(AA)):
        Pfiles[a][j]=AA[j]
    a+=1
    f.close()



delta=-0.5
dt1=0.001
dt2=0.1
tint=0
tmid=2
tmax=10
tt1=np.arange(tint,tmid,dt1)
tt2=np.arange(tmid,tmax+dt2,dt2)
tt=np.concatenate((tt1,tt2), axis=0)   
cwd = os.getcwd() #current working directory
item2=glob.glob("*.txt")
Qfiles=np.zeros((len(item2),len(tt)))
a=0
for i in range(len(item2)):
    f=open(item2[i],'r')
    AA=f.readlines()
    for j in range(len(AA)):
        Qfiles[a][j]=AA[j]
    a+=1
    f.close()

plt.plot(tm,Pfiles[0],label='PBC')
for i in range(len(item2)):
    plt.plot(tt,Qfiles[i],label=item2[i][:-4])
plt.legend(loc='upper right')
plt.show()

dy = np.zeros(Qfiles.shape,np.float)
dy[:,0:-1] = np.diff(Qfiles)/np.diff(tt)
dy[:,-1] = (Qfiles[:,-1] - Qfiles[:,-2])/(tt[-1] - tt[-2])

dy2 = np.zeros(Pfiles.shape,np.float)
dy2[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy2[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])

plt.plot(tm[1000:1500],dy2[0][1000:1500],label='PBC')
for i in range(len(item2)):
    plt.plot(tt[1000:1500],dy[i][1000:1500],label=item2[i][:-4])
plt.legend(loc='upper right')
plt.show()

#plt.plot(tt,Qfiles[0],'r',label='PBC')
#plt.plot(tt,Qfiles[2],'b',label='W=0.1')
#for i in range(1,len(Qfiles)):
 #   plt.plot(tt,Qfiles[i],label=item2[i][19:-4])
#plt.legend(loc='best')
#plt.xlabel('Time t')
#plt.ylabel('Return rate l(t)')
#plt.xlim(1.1,1.5)
#plt.show()