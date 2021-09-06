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

delta=0.5
dt1=0.01
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
item=item[::-1]
tm=t
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


"""Plot the Figure """
fig, ax = plt.subplots()
for num in range(len(Pfiles)):
    plt.plot(tm,Pfiles[num],label=item[num][23:-10])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
plt.legend(loc='upper right')#bbox_to_anchor=(1.28, 1.02),ncol=1)
plt.ylim(-0.05,0.8)
plt.xlim(-0.5,13.5)
iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax, [0.25, 0.5, 0.45, 0.45]) #posx, posy, width, height
iax.set_axes_locator(ip)
for num in range(len(Pfiles)):
    plt.plot(tm,dy[num])#label=item[num][27:-10])
plt.xlim([0,2])
plt.xticks([])
plt.yticks([])
plt.show()


fig, ax = plt.subplots()
plt.plot(tm,dy[-1],label=item[-1][27:-10])
for num in range(1,len(Pfiles)-1):
    plt.plot(tm,dy[num],label=item[num][27:-10])
plt.plot(tm,dy[0],label=item[0][27:-10])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
plt.legend(bbox_to_anchor=(1.28, 1.02),ncol=1)
#plt.ylim(-0.05,0.8)
iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax, [0.45, 0.5, 0.45, 0.45]) #posx, posy, width, height
iax.set_axes_locator(ip)
for num in range(len(Pfiles)-1):
    plt.plot(tm,dy[num],label=item[num][27:-10])
plt.xlim([0,2])
plt.xticks([])
plt.yticks([])
plt.show()



"""
item2=glob.glob("*.dat")
tm=t
qf=np.zeros((len(item2),len(tm)))
a=0
for i in range(len(item2)):
    f=open(item2[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        qf[a][j]=AA[j]
    a+=1

fig, ax = plt.subplots()
for num in range(len(qf)):
    plt.plot(tm,qf[num],label=item2[num][27:-10])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
plt.legend(bbox_to_anchor=(1.24, 1.02),ncol=1)
plt.ylim(-0.05,0.8)

iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax, [0.45, 0.5, 0.45, 0.45]) #posx, posy, width, height
iax.set_axes_locator(ip)
for num in range(len(qf)):
    plt.plot(tm,qf[num],label=item2[num][27:-10])
plt.xlim([1,1.4])
plt.ylim([0.4,0.55])
plt.show()
"""



