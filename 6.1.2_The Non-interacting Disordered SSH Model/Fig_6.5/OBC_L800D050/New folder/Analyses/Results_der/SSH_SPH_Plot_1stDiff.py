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
from matplotlib.patches import Rectangle

L=800
delta=-0.5
dt1=0.001
dt2=0.1
tint=0
tmid=2
tmax=10
t1=np.arange(tint,tmid,dt1)
t2=np.arange(tmid,tmax+dt2,dt2)
tm=np.concatenate((t1,t2), axis=0)
cwd = os.getcwd() #current working directory
item=glob.glob("*.txt")
Pfiles=np.zeros((len(item),len(tm)))
a=0
for i in range(len(item)):
    f=open(item[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        Pfiles[a][j]=AA[j]
    a+=1



dy = np.zeros(Pfiles.shape,np.float)
dy[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])


"""Plots of the Data """
Wlabel=['5e-08','5e-07','5e-06','5e-05','5e-04','5e-03','1e-02','5e-02','5e-14']
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm,Pfiles[-1],label='W='+Wlabel[-1])
for num in range(len(Pfiles)-1):
    if num < 8:
        plt.plot(tm,Pfiles[num],label='W='+Wlabel[num])
    elif num >= 8:
        plt.plot(tm,Pfiles[num],':',label=item[num][19:-4])
    else:
        plt.plot(tm,Pfiles[num],label=item[num][19:-4])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1)
plt.show()


tc1=np.zeros(len(item))
for i in range(len(tc1)):
    tc1[i]=np.amax(Pfiles[i])
    for j in range(len(Pfiles[i])):
        if tc1[i]==Pfiles[i][j]:
            print(j)
            tc1[i]=tm[int(j)]

fig = plt.figure(figsize=(6,4))
ax = plt.subplot(111)
ax.plot(tm[1200:1360],dy[-1][1200:1360],label='W='+Wlabel[-1])
for num in range(len(Pfiles)-1):
    if num < 8:
        ax.plot(tm[1200:1360],dy[num][1200:1360],label='W='+Wlabel[num])
    elif num >= 8:
        ax.plot(tm[1200:1360],dy[num][1200:1360],':',label=item[num][19:-4])
    else:
        ax.plot(tm[1200:1360],dy[num][1200:1360],label=item[num][19:-4])                
for i in range(0,5):
    plt.axvline(tc1[i],color=colors['grey'],linestyle='--')
plt.xlabel('t')
plt.ylabel(r'dl$^{\ast}$(t)/dt')
plt.ylim(-1.1,0.7)
ax.legend(bbox_to_anchor=(1.0, 1.02),ncol=1)
plt.show()


