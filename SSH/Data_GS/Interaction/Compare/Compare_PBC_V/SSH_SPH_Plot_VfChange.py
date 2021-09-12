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

L=10
V=0.1
BC='PBC, '
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

"""Plot the Figure """
#tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' and $\mu\in$ [$-W,W$]'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

plt.figure(figsize=(10, 6))
#fig=plt.figure(1, figsize=(3, 3))
ax = plt.subplot(221)
plt.ylabel('l(t)')
plt.xlabel('t')
textr='\n'.join((
        r'V: 0.01'+' -> '+'0.01',))
plt.plot(tm,Qfiles[0],label=r'$\Delta$: 0.01 -> 0.01')
plt.plot(tm,Pfiles[0],label=r'$\Delta$: 0      -> 0.01')
plt.ylim(-0.1,1.5)
#plt.text(8.0,2.3,textr,fontsize='9',bbox=props)
plt.text(11,1.3,'(A)')
plt.legend(loc='upper right',fontsize='8')

ax1 = plt.subplot(222)
textr='\n'.join((
        r'V: 0.1'+' -> '+'0.1',))
plt.plot(tm,Qfiles[1],label=r'$\Delta$: 0.1 -> 0.1')
plt.plot(tm,Pfiles[1],label=r'$\Delta$: 0    -> 0.1')
#plt.ylim(-0.1,1.2)
#plt.text(8.0,1.81,textr,fontsize='9',bbox=props)
plt.text(12,1.02,'(B)')
plt.ylabel('l(t)')
plt.xlabel('t')
plt.legend(loc='upper right',fontsize='8')

ax2 = plt.subplot(223)
textr='\n'.join((
        r'V: 1.0'+' -> '+'1.0',))
plt.plot(tm,Qfiles[2],label=r'$\Delta$: 3.0 -> 3.0')
plt.plot(tm,Pfiles[2],label=r'$\Delta$: 0 -> 3.0')
plt.ylim(-0.05,1.0)
#plt.text(8.0,1.2,textr,fontsize='9',bbox=props)
plt.text(12,.85,'(C)')
plt.ylabel('l(t)')
plt.xlabel('t')
plt.legend(loc='upper right',fontsize='8')


ax2 = plt.subplot(224)
textr='\n'.join((
        r'V: 8.0'+' -> '+'8.0',))
plt.plot(tm,Qfiles[3],label=r'$\Delta$: 8.0 -> 8.0')
plt.plot(tm,Pfiles[3],label=r'$\Delta$: 0 -> 8.0')
#plt.ylim(-0.1,1.6)
#plt.text(8.0,1.2,textr,fontsize='9',bbox=props)
plt.text(12,1.01,'(D)')#
plt.ylabel('l(t)')
plt.xlabel('t')
plt.legend(loc='upper right',fontsize='8')
