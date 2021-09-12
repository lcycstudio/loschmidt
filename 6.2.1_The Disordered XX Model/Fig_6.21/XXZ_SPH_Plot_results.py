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
dt1=0.001
dt2=0.1
tint=0
tmid=3
tmax=10
t1=np.arange(tint,tmid,dt1)
t2=np.arange(tmid,tmax+dt2,dt2)
t=np.concatenate((t1,t2), axis=0)
#dt2=0.05
cwd = os.getcwd() #current working directory
item=glob.glob("*.txt")
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
item2=glob.glob("*.dat")
qf=np.zeros((len(item2),len(tm)))
a=0
for i in range(len(item2)):
    f=open(item2[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        qf[a][j]=AA[j]
    a+=1


AS=qf[0]
"""Plot the Figure """
#tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' and $\mu\in$ [$-W,W$]'

wb=['0.0','5e-05','1e-04','5e-04','1e-03','5e-03','1e-02','5e-02','0.1','0.5','1.0']
fig = plt.figure()
ax = plt.subplot(111)
plt.plot(tm,AS,label='Analytic')
plt.plot(tm,Pfiles[0],label='W=0(ED)')
for num in range(1,len(Pfiles)):
    if num < 5:
        ax.plot(tm,Pfiles[num],label='W='+wb[num])
    else:
        ax.plot(tm,Pfiles[num],'--',label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'l$^\ast$(t)')
plt.xlim(-0.5,14)
ax.legend(loc='upper right',fontsize='9')
plt.text(0.05,1.4,'(A)')
plt.show()

dy = np.zeros(Pfiles.shape,np.float)
dy[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])
dyy=np.zeros(AS.shape,np.float)
dyy[0:-1]=np.diff(AS)/np.diff(tm)
dyy[-1]=(AS[-1]-AS[-2])/(tm[-1]-tm[-2])



plt.plot(tm,AS,label='Analytic')
plt.plot(tm,dy[0],label='W=0(ED)')
for num in range(1,len(Pfiles)):
    if num < 5:
        plt.plot(tm,dy[num],label='W='+wb[num])
    else:
        plt.plot(tm,dy[num],':',label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'dl$^\ast$/dt')
plt.legend(loc='upper right',fontsize='9')
plt.text(0.05,55,'(B)')
iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax, [0.27, 0.5, 0.45, 0.45]) #posx, posy, width, height
iax.set_axes_locator(ip)
plt.plot(tm[1560:1580],AS[1560:1580])
for num in range(len(Pfiles)):
    if num < 5:
        plt.plot(tm[1560:1580],dy[num][1560:1580])#,label='W='+wb[num])
    else:
        plt.plot(tm[1560:1580],dy[num][1560:1580],':')#,'--',label='W='+wb[num])

plt.show()

"""
for num in range(len(Pfiles)):
    if num < 5:
        plt.plot(tm[1560:1600],dy[num][1560:1600],label='W='+wb[num])
    else:
        plt.plot(tm[1560:1600],dy[num][1560:1600],'--',label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'dl$^\ast$/dt')
plt.legend(loc='upper right',fontsize='9')
plt.show()
    

plt.plot(tm[1573:1625],AS[1573:1625],label='Analytic')
plt.plot(tm[1573:1625],Pfiles[0][1573:1625],label='W=0(ED)')
for num in range(1,7):
    if num < 5:
        plt.plot(tm[1573:1625],Pfiles[num][1573:1625],label='W='+wb[num])
    else:
        plt.plot(tm[1573:1625],Pfiles[num][1573:1625],':',label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'l$^\ast$(t)')
plt.legend(loc='upper right',fontsize='9')
plt.show()
"""

plt.plot(tm[2000:2050],AS[2000:2050],label='Analytic')
plt.plot(tm[2000:2050],Pfiles[0][2000:2050],label='W=0(ED)')
for num in range(1,7):
    if num < 5:
        plt.plot(tm[2000:2050],Pfiles[num][2000:2050],label='W='+wb[num])
    else:
        plt.plot(tm[2000:2050],Pfiles[num][2000:2050],':',label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'l$^\ast$(t)')
plt.legend(loc='upper right',fontsize='9')
plt.text(2.015,1.1,'(C)')
plt.show()

"""
for num in range(7):
    if num < 5:
        plt.plot(tm[1573:1625],dy[num][1573:1625],label='W='+wb[num])
    else:
        plt.plot(tm[1573:1625],dy[num][1573:1625],'--',label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'dl$^\ast$/dt')
plt.legend(loc='upper right',fontsize='9')
plt.show()



fig=plt.figure(1,figsize=(8,5))
ax1=plt.subplot(221)
for num in range(7):
    if num < 5:
        plt.plot(tm[1573:1925],Pfiles[num][1573:1925],label='W='+wb[num])
    else:
        plt.plot(tm[1573:1925],Pfiles[num][1573:1925],':',label='W='+wb[num])
ax2=plt.subplot(222)
for num in range(7):
    if num < 5:
        plt.plot(tm[1925:2275],Pfiles[num][1925:2275],label='W='+wb[num])
    else:
        plt.plot(tm[1925:2275],Pfiles[num][1925:2275],':',label='W='+wb[num])
ax3=plt.subplot(223)
for num in range(7):
    if num < 5:
        plt.plot(tm[2275:2625],Pfiles[num][2275:2625],label='W='+wb[num])
    else:
        plt.plot(tm[2275:2625],Pfiles[num][2275:2625],':',label='W='+wb[num])
ax4=plt.subplot(224)
for num in range(7):
    if num < 5:
        plt.plot(tm[2625:3000],Pfiles[num][2625:3000],label='W='+wb[num])
    else:
        plt.plot(tm[2625:3000],Pfiles[num][2625:3000],':',label='W='+wb[num])
#plt.xlabel('t')
#plt.ylabel(r'l$^\ast$(t)')
#ax.legend(loc='upper right',fontsize='9')
plt.show()
"""