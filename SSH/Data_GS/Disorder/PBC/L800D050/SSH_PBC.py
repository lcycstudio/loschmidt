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

delta=-0.5
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
item=glob.glob("*.txt")
item2=glob.glob("*.dat")
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
qf=np.zeros((len(item2),len(tm)))
a=0
for i in range(len(item2)):
    f=open(item2[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        qf[a][j]=AA[j]
    a+=1


fig1=plt.figure(figsize=(12, 4))

ax1 = plt.subplot(121)

plt.plot(tm,qf[0],label=item2[0][27:-10])
for num in range(1,4):
    plt.plot(tm,Pfiles[num],label=item[num][27:-10])
plt.plot(tm,Pfiles[0],label=item[0][27:-10])
plt.plot(tm,Pfiles[6],label=item[6][27:-10])
plt.plot(tm,Pfiles[5],label=item[5][27:-10])
plt.plot(tm,Pfiles[4],label=item[4][27:-10])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
plt.legend(loc='upper right',fontsize='9')#bbox_to_anchor=(1.28, 1.02),ncol=1)
ymin=-0.03
ymax=0.6
top=np.amax(Pfiles[-1])
plt.ylim(ymin,ymax)
#plt.xlim(-0.5,12)
plt.text(0.05,0.5,'(C)')
plt.axvline(1.24,ymin,(top-ymin)/(ymax-ymin),linestyle=':')
plt.text(1.24+0.1,ymin+(0-ymin)/2,r'$t_{c_1}$')

iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax1, [0.3, 0.55, 0.4, 0.4]) #posx, posy, width, height
iax.set_axes_locator(ip)
plt.plot(tm,Pfiles[-1],label=item[-1][27:-10])
for num in range(2,len(Pfiles)-1):
    plt.plot(tm,Pfiles[num],label=item[num][27:-10])
plt.plot(tm,Pfiles[1],label=item[1][27:-10])
plt.plot(tm,Pfiles[0],label=item[0][27:-10])
#plt.xlim([0.9,1.5])
#plt.ylim([0.4,0.55])
#plt.show()



ax2= plt.subplot(122)
dy = np.zeros(Pfiles.shape,np.float)
dy[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])

dy2 = np.zeros(qf.shape,np.float)
dy2[:,0:-1] = np.diff(qf)/np.diff(tm)
dy2[:,-1] = (qf[:,-1] - qf[:,-2])/(tm[-1] - tm[-2])

plt.plot(tm,dy2[0],label=item2[0][27:-10])
for num in range(1,4):
    plt.plot(tm,dy[num],label=item[num][27:-10])
plt.plot(tm,dy[0],label=item[0][27:-10])
plt.plot(tm,dy[6],label=item[6][27:-10])
plt.plot(tm,dy[5],label=item[5][27:-10])
plt.plot(tm,dy[4],label=item[4][27:-10])
plt.xlabel('t')
plt.ylabel(r'dl$^{\ast}$/dt')
#plt.legend(loc='upper right',fontsize='9')#bbox_to_anchor=(1.28, 1.02),ncol=1)
#plt.xlim(-0.5,13.5)
#plt.ylim(-1.3,1.0)
#plt.text(0.0,0.7,'(A2)')
iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax2, [0.45, 0.05, 0.4, 0.4]) #posx, posy, width, height
iax.set_axes_locator(ip)

plt.plot(tm,dy2[0],label=item2[0][27:-10])
for num in range(1,4):
    plt.plot(tm,dy[num],label=item[num][27:-10])
plt.plot(tm,dy[0],label=item[0][27:-10])
plt.plot(tm,dy[6],label=item[6][27:-10])
plt.plot(tm,dy[5],label=item[5][27:-10])
plt.plot(tm,dy[4],label=item[4][27:-10])
plt.xlim([0.8,1.5])
plt.xticks([])
plt.yticks([])
plt.show()





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

fig2=plt.figure(figsize=(12, 4))

ax3 = plt.subplot(121)

for num in range(len(qf)):
    plt.plot(tm,qf[num],label=item2[num][27:-10])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
plt.legend(loc='upper right',fontsize='9')#bbox_to_anchor=(1.24, 1.02),ncol=1)
plt.ylim(-0.03,0.6)
#plt.xlim(-0.5,12)
plt.text(0.0,0.5,'(D)')
iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax3, [0.3, 0.55, 0.4, 0.4]) #posx, posy, width, height
iax.set_axes_locator(ip)
for num in range(len(qf)):
    plt.plot(tm,qf[num],label=item2[num][27:-10])
plt.xlim([1,1.42])
plt.ylim([0.4,0.55])

qy = np.zeros(qf.shape,np.float)
qy[:,0:-1] = np.diff(qf)/np.diff(tm)
qy[:,-1] = (qf[:,-1] - qf[:,-2])/(tm[-1] - tm[-2])

ax4 = plt.subplot(122)

for num in range(len(qf)):
    plt.plot(tm,qy[num],label=item2[num][27:-10])
plt.xlabel('t')
plt.ylabel(r'dl$^{\ast}$/dt')
plt.legend(loc='upper right',fontsize='9')#bbox_to_anchor=(1.28, 1.02),ncol=1)
plt.xlim(-0.5,13.5)
plt.ylim(-1.3,1.0)
#plt.text(0.0,0.7,'(B2)')
iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax4, [0.4, 0.05, 0.4, 0.4]) #posx, posy, width, height
iax.set_axes_locator(ip)
for num in range(len(qf)):
    plt.plot(tm,qy[num],label=item2[num][27:-10])
plt.xlim([1.1,1.4])
plt.xticks([])
plt.yticks([])
plt.show()


