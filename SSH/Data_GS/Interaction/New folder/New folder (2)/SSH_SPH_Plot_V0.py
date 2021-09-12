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
V=0
delta=-0.95
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
textr='\n'.join((
        r'V='+str(V),
        r'$\delta: -0.95$'+' -> '+r'$+0.95$'))
fig = plt.figure()
ax = plt.subplot(111)
#ax.plot(tm,-2*np.log(Qfiles[0])/L,':',label=r'V='+item[0][13:16])
for num in range(len(Pfiles)):
    ax.plot(tm,Pfiles[num],label=item[num][:-4])#[13:17])
plt.xlabel('t')
plt.ylabel('l(t)')
#plt.ylim(-0.05,1.0)
plt.text(10,3.0,textr,bbox=props)
ax.legend(bbox_to_anchor=(1.0, 1.02),ncol=1)
plt.show()

for i in range(0,len(item),2):
    plt.plot(tm,Pfiles[i],label=item[i][:-4])
plt.legend(loc='best')#bbox_to_anchor=(1.0, 1.02),ncol=1)
plt.show()

for i in range(1,len(item),2):
    plt.plot(tm,Pfiles[i],label=item[i][:-4])
plt.legend(loc='best')#bbox_to_anchor=(1.0, 1.02),ncol=1)
plt.show()

