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

V=0
BC='PBC, '
delta=-0.95
delta2=-delta
dt1=0.01
dt2=0.01
dt3=0.01
tint=0
tmid1=2
tmid2=3
tmax=20
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
#tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' and $\mu\in$ [$-W,W$]'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

fig=plt.figure(1, figsize=(4, 5))
ax = fig.add_subplot(311)
textr='\n'.join((
        #r'V = '+str(V)+', W = 0',
        BC+r'$\delta: -0.15$'+' -> '+r'$+0.15$',))
plt.plot(tm,Pfiles[0])#,label=item[0][29:-4])
plt.plot(tm,Pfiles[1])#,label=item[1][29:-4])
#plt.xlabel('t')
#plt.ylabel('l(t)')
plt.ylim(-0.02,0.45)
plt.text(8.0,0.35,textr,bbox=props)
plt.legend(loc='upper right')
#plt.show()

ax1 = fig.add_subplot(312)
textr='\n'.join((
        #r'V = '+str(V)+', W = 0',
        BC+r'$\delta: -0.5$'+' -> '+r'$+0.5$',))
plt.plot(tm,Pfiles[2])#,label=item[2][28:-4])
plt.plot(tm,Pfiles[3])#,label=item[3][28:-4])
#plt.xlabel('t')
plt.ylabel('l(t)')
plt.ylim(-0.04,0.7)
plt.text(8.0,0.55,textr,bbox=props)
plt.legend(loc='upper right')
#plt.show()

ax2 = fig.add_subplot(313)
textr='\n'.join((
        #r'V = '+str(V)+', W = 0',
        BC+r'$\delta: -0.95$'+' -> '+r'$+0.95$',))
plt.plot(tm,Pfiles[4])#,label=item[4][29:-4])
plt.plot(tm,Pfiles[5])
plt.plot(tm,Pfiles[6])
plt.xlabel('t')
#plt.ylabel('l(t)')
plt.ylim(-0.05,0.95)
plt.text(8.0,0.75,textr,bbox=props)
plt.legend(loc='upper right')

ax1.set_ylabel('l(t)')
plt.show()

