# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:57:26 2018

@author: LewisCYC
"""

import os             
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
#from scipy.optimize import curve_fit

#from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
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

item2=glob.glob("*.dat")
Qfiles=np.zeros((len(item2),len(tm)))
a=0
for i in range(len(item2)):
    f=open(item2[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        Qfiles[a][j]=AA[j]
    a+=1
    

""" Smoothing out W=0.0 """
maxIndex=Pfiles[0].tolist().index(np.amax(Pfiles[0]))
tn=np.arange(maxIndex,2000,30)
tIndex=np.concatenate((tn,np.array([2000])), axis=0)
x=np.zeros(len(tIndex))
y=np.zeros(len(x))
for jj in range(len(x)):
    y[jj]=Pfiles[0][tIndex[jj]]
    x[jj]=tm[tIndex[jj]]
z=np.polyfit(x,y,2)
p=np.poly1d(z)
dt3=len(Pfiles[0][maxIndex:2000])
xp=np.linspace(x[0],x[-1],dt3)
yp=p(xp)
fig = plt.figure()
ax = plt.subplot(111)
Wlabel=['0.0','5e-7','5e-6','5e-5']
ax.plot(xp,p(xp),'k:',label='Polyfit')
Pfiles[0][maxIndex:2000]=yp
tm0=tm
tm0[maxIndex:2000]=xp
plt.show()


"""Plots of the Data """
Wlabel=['0.0','5e-8','5e-7','5e-6','5e-5','5e-4','5e-3','5e-2']
fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    if num <5:
        if num == 0:
            ax.plot(tm0[1000:1600],Pfiles[num][1000:1600],label='W='+Wlabel[num])
        else:
            ax.plot(tm[1000:1600],Pfiles[num][1000:1600],label='W='+Wlabel[num])
    elif num > 9:
        ax.plot(tm[1000:1600],Pfiles[num][1000:1600],':',label='W='+Wlabel[num])
    else:
        ax.plot(tm[1000:1600],Pfiles[num][1000:1600],label='W='+Wlabel[num])
for kk in range(len(item2)):
    ax.plot(tm[1000:1600],Qfiles[kk][1000:1600],'--',label=item2[kk][-11:-4])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
plt.legend(bbox_to_anchor=(1.0,1.02),ncol=1)
#plt.xlim(1.0,1.6)
#plt.ylim(0.25,0.6)
plt.show()


