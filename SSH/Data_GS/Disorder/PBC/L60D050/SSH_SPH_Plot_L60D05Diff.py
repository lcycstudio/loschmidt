# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:34:20 2017

@author: LewisCYC
"""
import os             
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

L=60
delta=0.5
dt1=0.01
dt2=0.01
dt3=0.01
tint=0
tmid1=2
tmid2=3
tmax=5
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


"""Plot the Figure """
Wlabel=['1e-4','1e-3','1e-2','0.1','0.3','0.5','0.8','1.0','1e-6','1e-8']
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textr='\n'.join((
        r'$\delta$: +'+str(delta)+' -> -'+str(delta),
        r'L = '+str(L)))

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm,dy2[0],label='W=0.0')
ax.plot(tm,dy[-1],label='W='+Wlabel[-1])
ax.plot(tm,dy[-2],label='W='+Wlabel[-2])
for num in range(len(Pfiles)-2):
    if num==len(item)-3:
        ax.plot(tm,dy[num],':',label='W='+Wlabel[num])
    else:
        ax.plot(tm,dy[num],label='W='+Wlabel[num])
plt.xlabel('t')
plt.ylabel(r'dl$^\ast$/dt')
plt.text(1.9,0.7,textr,bbox=props)
iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax, [0.35, 0.02, 0.35, 0.35]) #posx, posy, width, height
iax.set_axes_locator(ip)
plt.plot(tm[90:150],dy2[0][90:150],label='W='+Wlabel[0])
plt.plot(tm[90:150],dy[-1][90:150],label='W='+Wlabel[-1])
plt.plot(tm[90:150],dy[-2][90:150],label='W='+Wlabel[-2])
for num in range(len(Pfiles)-2):
    if num==len(item)-3:
        plt.plot(tm[90:150],dy[num][90:150],':',label='W='+Wlabel[num])
    else:
        plt.plot(tm[90:150],dy[num][90:150],label='W='+Wlabel[num])
plt.xticks([])
plt.yticks([])
plt.show()



fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm[20:200],dy2[0][20:200],label='W=0.0')
ax.plot(tm[20:200],dy[-1][20:200],label='W='+Wlabel[-1])
ax.plot(tm[20:200],dy[-2][20:200],label='W='+Wlabel[-2])
for num in range(len(Pfiles)-2):
    if num==len(item)-3:
        ax.plot(tm[20:200],dy[num][20:200],':',label='W='+Wlabel[num])
    else:
        ax.plot(tm[20:200],dy[num][20:200],label='W='+Wlabel[num])
plt.xlabel('t')
plt.ylabel(r'dl$^\ast$/dt')
#plt.text(0.6,4,textr,bbox=props)
plt.legend(loc='upper right')#bbox_to_anchor=(1.0, 1.02),ncol=1)
plt.show()
