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

L=12
V=3.0
delta=-0.95
dt1=0.005
tint=0
tmax=20
t=np.arange(tint,tmax+dt1/2,dt1)
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

#Pfiles[1:]=-2*np.log(Pfiles[1:])/L
"""Plot the Figure """
wb=['1e-4','1e-3','0.01','0.05','0.1','1.0','5.0']
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textr='\n'.join((
        r'(C1) $\Delta$: 0'+' -> '+str(V),))

plt.figure(figsize=(12, 4))
ax = plt.subplot(121)

ax.plot(tm,Qfiles[0],label='W=0.0')
for num in range(len(Pfiles)):
    ax.plot(tm,Pfiles[num],label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'l$^\ast$(t)')
plt.ylim(-0.05,1.3)

ymin=-0.05
ymax=0.8
plt.ylim(ymin,ymax)

plt.text(0.3,0.9*ymax,textr)
ax.legend(loc='upper right',fontsize='9')
tm[:200]
Qfiles[0][:200]
top=np.amax(Qfiles[0][:200])
for i in range(200):
    if top==Qfiles[0][i]:
        tc=tm[i]
plt.axvline(tc,ymin,top/ymax,linestyle=':')
textr2='\n'.join((
        r'$t_{c_1}$',))
plt.text(tc+0.1,ymin+0.025,textr2)#,bbox=props)


ax1 = plt.subplot(122)
dy = np.zeros(Pfiles.shape,np.float)
dy[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])
dy2 = np.zeros(Qfiles.shape,np.float)
dy2[:,0:-1] = np.diff(Qfiles)/np.diff(tm)
dy2[:,-1] = (Qfiles[:,-1] - Qfiles[:,-2])/(tm[-1] - tm[-2])

#plt.ylim(-35,35)
#plt.xlim(-1,20.5)
plt.plot(tm,dy2[0],label='W=0.0')
for num in range(len(Pfiles)):
    plt.plot(tm,dy[num],label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'dl$^\ast$/dt')
plt.text(0.05,2.0,'(C2)')


iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax1, [0.23, 0.03, 0.4, 0.4]) #posx, posy, width, height
iax.set_axes_locator(ip)
plt.plot(tm[0:200],dy2[0][0:200],label='W=0.0')
for num in range(len(Pfiles)):
    plt.plot(tm[0:200],dy[num][0:200],label='W='+wb[num])
plt.yticks([])
plt.xticks([])
plt.show()
print(r'$t_{c_1}$ is ',tc)
