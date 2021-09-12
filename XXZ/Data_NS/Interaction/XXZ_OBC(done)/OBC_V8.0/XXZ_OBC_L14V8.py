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

L=14
V=8.0
dt1=0.005
tint=0
tmax=50
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

"""Plot the Figure """
wb=['1e-4','1e-3','0.01','0.05','0.1','1.0','3.5','6.0']
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textr='\n'.join((
        r'(C) $\Delta$ = 8.0',))

fig = plt.figure(figsize=(12,4))
ax1 = plt.subplot(121)

ax1.plot(tm,Qfiles[0],label='W=0.0')
for num in range(len(Pfiles)-2):
    ax1.plot(tm,Pfiles[num],label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'l$^\ast$(t)')
plt.ylim(-0.001,0.019)
plt.text(0.3,0.0175,textr)
ax1.legend(loc='lower right',fontsize='9')
tm[:2000]
Qfiles[0][:2000]
top=np.amax(Qfiles[0][:2000])
for i in range(2000):
    if top==Qfiles[0][i]:
        tc=tm[i]
plt.axvline(tc,-0.5,top/0.0175,linestyle=':')
plt.text(tc+0.5,0.00,r'$t_{c_1}$')
print(r'tc is',tc)


dy = np.zeros(Pfiles.shape,np.float)
dy[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])

AS=Qfiles[0]
dyy=np.zeros(AS.shape,np.float)
dyy[0:-1]=np.diff(AS)/np.diff(tm)
dyy[-1]=(AS[-1]-AS[-2])/(tm[-1]-tm[-2])

ax2 = plt.subplot(122)
ax2.plot(tm,dyy,label='W=0.0')
for num in range(len(Pfiles)-2):
    ax2.plot(tm,dy[num],label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'dl$^\ast$/dt')
plt.show()
#plt.ylim(-0.001,0.019)
#plt.text(0.3,0.0175,textr)
#ax1.legend(loc='lower right',fontsize='9')



fig = plt.figure(figsize=(12,4))

ax3 = plt.subplot(121)
textr='\n'.join((
        r'(D) $\Delta$ = 8.0',))
ax3.plot(tm,Pfiles[4],label='W='+wb[4])
ax3.plot(tm,Pfiles[5],label='W='+wb[5])
ax3.plot(tm,Pfiles[6],label='W='+wb[6])
ax3.plot(tm,Pfiles[7],label='W='+wb[7])
plt.xlabel('t')
plt.ylabel(r'l$^\ast$(t)')
plt.ylim(-0.005,0.1)
plt.text(0.3,0.09,textr)
plt.legend(loc='upper right',fontsize='9')

ax4 = plt.subplot(122)

ax4.plot(tm,dy[4],label='W='+wb[4])
ax4.plot(tm,dy[5],label='W='+wb[5])
ax4.plot(tm,dy[6],label='W='+wb[6])
ax4.plot(tm,dy[7],label='W='+wb[7])
plt.show()
