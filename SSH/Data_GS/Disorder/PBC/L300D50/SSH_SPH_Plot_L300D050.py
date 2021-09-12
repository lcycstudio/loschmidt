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

L=300
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
t=np.concatenate((tt,t3), axis=0)
#dt2=0.05
cwd = os.getcwd() #current working directory
item=glob.glob("*.dat")
item2=glob.glob("*.txt")
tm=t
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

"""Plot the Figure """
Wlabel=['0.01','0.1','0.3','0.1','0.3','0.5','0.8','1.0','1e-6','1e-8']
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textr='\n'.join((
        r'(B) L = '+str(L)+', $|\delta|$= '+str(delta),))#+' -> '+str(-delta),))

plt.figure(figsize=(12, 4))

ax = plt.subplot(121)


ax.plot(tm,Qfiles[0],label='W=0.0')
for num in range(len(Pfiles)):
    ax.plot(tm,Pfiles[num],label=item[num][23:-10])
plt.xlabel('t')
plt.ylabel(r'l$^\ast$(t)')
#plt.xlim(-0.2,6.7)

ymin=-0.05
ymax=0.6
plt.ylim(ymin,ymax)
tm[:200]
Qfiles[0][:200]
top=np.amax(Qfiles[0][:200])
for i in range(200):
    if top==Qfiles[0][i]:
        tc=tm[i]
plt.axvline(tc,0,(top-ymin)/(ymax-ymin),linestyle=':')
textr2='\n'.join((
        r'$t_{c_1}$',))
plt.text(tc+0.1,ymin+(0-ymin)/2,textr2)
plt.text(0.05,0.9*(ymax-ymin)+ymin,'(B)')
plt.legend(loc='upper right',fontsize='9')

ax1 = plt.subplot(122)

dy = np.zeros(Pfiles.shape,np.float)
dy[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])

dy2 = np.zeros(Qfiles.shape,np.float)
dy2[:,0:-1] = np.diff(Qfiles)/np.diff(tm)
dy2[:,-1] = (Qfiles[:,-1] - Qfiles[:,-2])/(tm[-1] - tm[-2])


plt.plot(tm,dy2[0],label='W=0.0')
for num in range(len(Pfiles)):
    plt.plot(tm,dy[num],label=item[num][23:-10])
plt.xlabel('t')
plt.ylabel(r'dl$^\ast$/dt')
iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax1, [0.4, 0.02, 0.4, 0.4]) #posx, posy, width, height
iax.set_axes_locator(ip)
plt.plot(tm[115:135],dy2[0][115:135],label='W=0.0')
for num in range(len(Pfiles)):
    plt.plot(tm[115:135],dy[num][115:135],label=item[num][23:-10])
plt.yticks([])
plt.xticks([])
plt.show()






print(r'$t_{c_1}$ is ',tc)
plt.show()
