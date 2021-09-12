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


""" Smoothing out W=5e-14 """
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


maxIndex=Pfiles[-1].tolist().index(np.amax(Pfiles[-1]))
tn=np.arange(maxIndex,2000,30)
tIndex=np.concatenate((tn,np.array([2000])), axis=0)
x=np.zeros(len(tIndex))
y=np.zeros(len(x))
for jj in range(len(x)):
    y[jj]=Pfiles[-1][tIndex[jj]]
    x[jj]=tm[tIndex[jj]]
z=np.polyfit(x,y,2)
p=np.poly1d(z)
dt3=len(Pfiles[-1][maxIndex:2000])
xp=np.linspace(x[0],x[-1],dt3)
yp=p(xp)
fig = plt.figure()
ax = plt.subplot(111)
Wlabel=['0.0','5e-7','5e-6','5e-5']
ax.plot(xp,p(xp),'k:',label='Polyfit')
Pfiles[-1][maxIndex:2000]=yp

maxIndex=Pfiles[-2].tolist().index(np.amax(Pfiles[-2]))
tn=np.arange(maxIndex,2000,30)
tIndex=np.concatenate((tn,np.array([2000])), axis=0)
x=np.zeros(len(tIndex))
y=np.zeros(len(x))
for jj in range(len(x)):
    y[jj]=Pfiles[-2][tIndex[jj]]
    x[jj]=tm[tIndex[jj]]
z=np.polyfit(x,y,2)
p=np.poly1d(z)
dt3=len(Pfiles[-2][maxIndex:2000])
xp=np.linspace(x[0],x[-1],dt3)
yp=p(xp)
fig = plt.figure()
ax = plt.subplot(111)
Wlabel=['0.0','5e-7','5e-6','5e-5']
ax.plot(xp,p(xp),'k:',label='Polyfit')
Pfiles[-2][maxIndex:2000]=yp


maxIndex=Pfiles[-3].tolist().index(np.amax(Pfiles[-3]))
tn=np.arange(maxIndex,2000,30)
tIndex=np.concatenate((tn,np.array([2000])), axis=0)
x=np.zeros(len(tIndex))
y=np.zeros(len(x))
for jj in range(len(x)):
    y[jj]=Pfiles[-3][tIndex[jj]]
    x[jj]=tm[tIndex[jj]]
z=np.polyfit(x,y,2)
p=np.poly1d(z)
dt3=len(Pfiles[-3][maxIndex:2000])
xp=np.linspace(x[0],x[-1],dt3)
yp=p(xp)
fig = plt.figure()
ax = plt.subplot(111)
Wlabel=['0.0','5e-7','5e-6','5e-5']
ax.plot(xp,p(xp),'k:',label='Polyfit')
Pfiles[-3][maxIndex:2000]=yp


tm0=tm
tm0[maxIndex:2000]=xp
dy = np.zeros(Pfiles.shape,np.float)
dy[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])

"""Plots of the Data """
Wlabel=['0.0','5e-8','5e-7','5e-6','5e-5']
fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    if num <5:
        if num == 0:
            ax.plot(tm0,Pfiles[num],label='W='+Wlabel[num])
        else:
            ax.plot(tm,Pfiles[num],label='W='+Wlabel[num])
    elif num > 9:
        ax.plot(tm,Pfiles[num],':',label=item[num][19:-4])
    else:
        ax.plot(tm,Pfiles[num],label=item[num][19:-4])
plt.xlabel('Time t')
plt.ylabel('1st Derivative dl/dt')
ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1)
plt.show()


"""Plots of the 1st Derivative"""
fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    if num <5:
        if num == 0:
            ax.plot(tm0,dy[num],label='W='+Wlabel[num])
        else:
            ax.plot(tm,dy[num],label='W='+Wlabel[num])
    elif num > 9:
        ax.plot(tm,dy[num],':',label=item[num][19:-4])
    else:
        ax.plot(tm,dy[num],label=item[num][19:-4])
plt.xlabel('Time t')
plt.ylabel('1st Derivative dl/dt')
ax.legend(bbox_to_anchor=(1.01, 1.0),ncol=1)
plt.show()



fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    if num <5:
        if num == 0:
            ax.plot(tm0[1000:1400],dy[num][1000:1400],label='W='+Wlabel[num])
        else:
            ax.plot(tm[1000:1400],dy[num][1000:1400],label='W='+Wlabel[num])
    elif num > 9:
        ax.plot(tm[1000:1400],dy[num][1000:1400],':',label=item[num][19:-4])
    else:
        ax.plot(tm[1000:1400],dy[num][1000:1400],label=item[num][19:-4])
plt.xlabel('t')
plt.ylabel(r'dl$^{\ast}$/dt')
plt.xlim([1.15,1.40])
ax.legend(bbox_to_anchor=(1.02, 1.02),ncol=1)
plt.show()

tc1=np.zeros(len(item))
for i in range(len(tc1)):
    tc1[i]=np.amax(Pfiles[i])
    for j in range(len(Pfiles[i])):
        if tc1[i]==Pfiles[i][j]:
            print(j)
            tc1[i]=tm[int(j)]


#fig = plt.figure(figsize=(6,4))
#ax = plt.subplot(111)
WWlabel=['5e-14','5e-15']
plt.plot(tm0,dy[0],label='W=0.0')
for num in range(1,len(Pfiles)):
    plt.plot(tm,dy[num],label='W='+WWlabel[num-1])
for i in range(len(tc1)):
    plt.axvline(tc1[i],color=colors['grey'],linestyle='--')
plt.legend(loc='best')
plt.xlim(1.3,1.45)
plt.xlabel('t')
plt.ylabel(r'dl$^{\ast}(t)$/dt')
plt.show()



