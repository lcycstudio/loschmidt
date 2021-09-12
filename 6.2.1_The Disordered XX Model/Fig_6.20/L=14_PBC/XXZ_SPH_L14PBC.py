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


dt2=0.01
tint=0
tmax=20
t=np.arange(tint,tmax+dt2,dt2)
#dt2=0.05
cwd = os.getcwd() #current working directory
item=glob.glob("*.dat")
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
item2=glob.glob("*.txt")
qf=np.zeros((len(item2),len(tm)))
a=0
for i in range(len(item2)):
    f=open(item2[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        qf[a][j]=AA[j]
    a+=1


AS=-2*np.log(qf[0])/14
Pfiles=-2*np.log(Pfiles)/14
"""Plot the Figure """
#tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' and $\mu\in$ [$-W,W$]'

wb=['1e-4','1e-3','0.01','0.05','0.1','0.5','1.0','3.5','6.0']
fig = plt.figure(figsize=(12,4))
ax1 = plt.subplot(121)
ax1.plot(tm,AS,label='W=0.0')
for num in range(len(Pfiles)):
    ax1.plot(tm,Pfiles[num],label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'l$^\ast$(t)')
plt.xlim(-0.5,24)
plt.legend(loc='upper right',fontsize='9')
plt.axvline(np.pi/2,linestyle=':')
plt.text(np.pi/2+0.1,-0.05,r'$t_{c_1}$')



dy = np.zeros(Pfiles.shape,np.float)
dy[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])
dyy=np.zeros(AS.shape,np.float)
dyy[0:-1]=np.diff(AS)/np.diff(tm)
dyy[-1]=(AS[-1]-AS[-2])/(tm[-1]-tm[-2])

ax2 = plt.subplot(122)
ax2.plot(tm,AS,label='W=0.0')
for num in range(len(Pfiles)):
    ax2.plot(tm,dy[num],label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'dl$^\ast$/dt')
plt.axvline(np.pi/2,linestyle=':')
plt.text(np.pi/2+0.1,-100,r'$t_{c_1}$')
#plt.legend(loc='upper right',fontsize='9')
plt.show()
#plt.text(0.05,55,'(B)')
"""
iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax2, [0.27, 0.5, 0.45, 0.45]) #posx, posy, width, height
iax.set_axes_locator(ip)
plt.plot(tm[1560:1580],AS[1560:1580])
for num in range(len(Pfiles)):
    if num < 5:
        plt.plot(tm[1560:1580],dy[num][1560:1580])#,label='W='+wb[num])
    else:
        plt.plot(tm[1560:1580],dy[num][1560:1580],':')#,'--',label='W='+wb[num])
"""

