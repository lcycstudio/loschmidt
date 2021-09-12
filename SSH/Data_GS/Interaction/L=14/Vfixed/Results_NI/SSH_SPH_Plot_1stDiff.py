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

L=14
delta=-0.5
W=0.5
V=1.0
tint=0
tmax=50
dt=0.2
tm=np.arange(tint,tmax+dt/2,dt)

cwd = os.getcwd() #current working directory
item=glob.glob("*.dat")
Pfiles=np.zeros((len(item),len(tm)))
a=0
for i in range(len(item)):
    f=open(item[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        Pfiles[a][j]=AA[j]
    a+=1


""" Smoothing out W=0.0 """
dy = np.zeros(Pfiles.shape,np.float)
dy[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])


"""Plots of the Data """
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm,Pfiles[7],label=item[7][:-4])
for num in range(len(Pfiles)-1):
    if num==2:
        ax.plot(tm,Pfiles[num],'--',label=item[num][26:-9])
    else:
        ax.plot(tm,Pfiles[num],label=item[num][26:-9])
plt.xlabel('Time t')
plt.ylabel('1st Derivative dl/dt')
ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1)
plt.show()


"""Plots of the 1st Derivative"""
Wlabel=['0.0001','0.001','0.01','0.1','0.5','1.0']
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm[150:230],dy[7][150:230],label=item[7][:-4])
for num in range(len(Pfiles)-1):
    if num==2:
        ax.plot(tm[150:230],dy[num][150:230],'--',label=item[num][26:-9])
    else:
        ax.plot(tm[150:230],dy[num][150:230],label=item[num][26:-9])
plt.xlabel('Time t')
plt.ylabel('1st Derivative dl/dt')
ax.legend(bbox_to_anchor=(1.02, 1.0),ncol=1)
plt.show()

Wlabel=['0.0001','0.001','0.01','0.1','0.5','1.0']
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm,dy[7],label=item[7][:-4])
for num in range(len(Pfiles)-1):
    if num==2:
        ax.plot(tm,dy[num],'--',label=item[num][26:-9])
    else:
        ax.plot(tm,dy[num],label=item[num][26:-9])
plt.xlabel('Time t')
plt.ylabel('1st Derivative dl/dt')
ax.legend(bbox_to_anchor=(1.02, 1.0),ncol=1)
plt.show()

