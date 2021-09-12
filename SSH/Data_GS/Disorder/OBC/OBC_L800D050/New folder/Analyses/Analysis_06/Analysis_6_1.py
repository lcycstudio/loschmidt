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
dt1=0.1
dt2=0.1
tint=0
tmid=2
tmax=100
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

plt.plot(tm,Pfiles[1],'g--',label='L=800(W=1e-5)')
plt.plot(tm,Pfiles[0],'orange',label='L=2400(PBC)')
plt.legend(bbox_to_anchor=(1.38,1.02),ncol=1)
plt.xlabel('t')
plt.ylabel('l(t)')
plt.xlim(-1,50)
plt.show()

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


""" Smoothing out W=0.0 """
maxIndex=Qfiles[0].tolist().index(np.amax(Qfiles[0]))
tn=np.arange(maxIndex,2000,30)
tIndex=np.concatenate((tn,np.array([2000])), axis=0)
x=np.zeros(len(tIndex))
y=np.zeros(len(x))
for jj in range(len(x)):
    y[jj]=Qfiles[0][tIndex[jj]]
    x[jj]=tt[tIndex[jj]]
z=np.polyfit(x,y,2)
p=np.poly1d(z)
dt3=len(Qfiles[0][maxIndex:2000])
xp=np.linspace(x[0],x[-1],dt3)
yp=p(xp)
fig = plt.figure()
ax = plt.subplot(111)
Wlabel=['0.0','5e-7','5e-6','5e-5']
#ax.plot(xp,p(xp),'k:',label='Polyfit')
Qfiles[0][maxIndex:2000]=yp
tm0=tt
tm0[maxIndex:2000]=xp



plt.plot(tm0,Qfiles[0],label='L=800(OBC)')
plt.plot(tm,Pfiles[0],label='L=2400(OBC)')
plt.plot(tt,Qfiles[1],'--',label='L=800(W=1e-6))')
plt.plot(tt,Qfiles[2],'--',label='L=800(W=0.1)')
plt.plot(tt,Qfiles[3],'--',label='L=800(W=0.25)')
plt.plot(tm,Pfiles[1],label='L=800(PBC)')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('l(t)')
plt.ylim(-0.05,0.75)
plt.show()

tc1=np.zeros(len(item))
for i in range(len(tc1)):
    tc1[i]=np.amax(Pfiles[i])
    for j in range(len(Pfiles[i])):
        if tc1[i]==Pfiles[i][j]:
            print(j)
            tc1[i]=tm[int(j)]

tc2=np.zeros(len(item2))
for i in range(len(tc2)):
    tc2[i]=np.amax(Qfiles[i])
    for j in range(len(Qfiles[i])):
        if tc2[i]==Qfiles[i][j]:
            print(j)
            tc2[i]=tt[int(j)]

plt.plot(tm0,Qfiles[0],label='L=800(OBC)')
plt.plot(tm,Pfiles[0],label='L=2400(OBC)')
plt.plot(tt,Qfiles[1],'--',label='L=800(W=1e-6))')
plt.plot(tt,Qfiles[2],'--',label='L=800(W=0.1)')
plt.plot(tt,Qfiles[3],'--',label='L=800(W=0.25)')
plt.plot(tm,Pfiles[1],label='L=800(PBC)')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('l(t)')
plt.xlim(1,2)
plt.ylim(0.2,0.6)
plt.show()

dy = np.zeros(Qfiles.shape,np.float)
dy[:,0:-1] = np.diff(Qfiles)/np.diff(tt)
dy[:,-1] = (Qfiles[:,-1] - Qfiles[:,-2])/(tt[-1] - tt[-2])

dy2 = np.zeros(Pfiles.shape,np.float)
dy2[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy2[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])


plt.plot(tt,dy[0],label='L=800(OBC)')
plt.plot(tm,dy2[0],label='L=2400(OBC)')
plt.plot(tt,dy[1],label='L=800(W=1e-5)')
plt.plot(tt,dy[2],label='L=800(W=0.1)')
plt.plot(tt,dy[3],label='L=800(W=0.25)')
plt.plot(tm,dy2[1],label='L=800(PBC)')
for i in range(len(tc1)):
    plt.axvline(tc1[i],color=colors['grey'],linestyle='--')
for i in range(len(tc2)):
    plt.axvline(tc2[i],color=colors['grey'],linestyle='--')
plt.xticks([tc1[0],tc1[1],tc2[0]])
plt.xlim(1.2,1.4)
plt.ylim(-2,4)
plt.xlabel('t')
plt.ylabel('dl(t)/dt')
plt.legend(loc='upper left')


#plt.plot(tt,Qfiles[0],'r',label='PBC')
#plt.plot(tt,Qfiles[2],'b',label='W=0.1')
#for i in range(1,len(Qfiles)):
 #   plt.plot(tt,Qfiles[i],label=item2[i][19:-4])
#plt.legend(loc='best')
#plt.xlabel('Time t')
#plt.ylabel('Return rate l(t)')
#plt.xlim(1.1,1.5)
#plt.show()