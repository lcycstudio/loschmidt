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

delta=-0.5
dt1=0.001
dt2=0.01
tint=0
tmid=4
tmax=10
t1=np.arange(tint,tmid,dt1)
t2=np.arange(tmid,tmax+dt2,dt2)
t=np.concatenate((t1,t2), axis=0)
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


for cc in range(len(item)):
    plt.plot(tm,Pfiles[cc])
plt.show()


"""Plot the Figure """
#tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' and $\mu\in$ [$-W,W$]'

Wlabel=['300','60','600','800']
fig = plt.figure()
ax = plt.subplot(111)
plt.plot(tm,Pfiles[1],label='W='+Wlabel[1])
plt.plot(tm,Pfiles[0],label='W='+Wlabel[0])
plt.plot(tm,Pfiles[2],label='W='+Wlabel[2])
plt.plot(tm,Pfiles[3],label='W='+Wlabel[3])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
ax.legend(bbox_to_anchor=(1.0, 1.02),ncol=1)
plt.show()

fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    if num <5:
        ax.plot(tm[500:2006],Pfiles[num][500:2006],label='W='+Wlabel[num])
    elif num > 9:
        ax.plot(tm[500:2006],Pfiles[num][500:2006],':',label=item[num][19:-4])
    else:
        ax.plot(tm[500:2006],Pfiles[num][500:2006],label=item[num][19:-4])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
ax.legend(bbox_to_anchor=(1.01, 1.0),ncol=1)
plt.show()


fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm[1000:1400],Pfiles[0][1000:1400],label='W='+Wlabel[0])
for num in range(1,len(Pfiles)):
    if num < 8:
        ax.plot(tm[1000:1400],Pfiles[num][1000:1400],label='W='+Wlabel[num])
    elif num > 9:
        ax.plot(tm[1000:1400],Pfiles[num][1000:1400],':',label=item[num][19:-4])
    else:
        ax.plot(tm[1000:1400],Pfiles[num][1000:1400],label=item[num][19:-4])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
plt.xlim([1.1,1.4])
ax.legend(bbox_to_anchor=(1.0, 1.02),ncol=1)
plt.show()


fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm[1200:1341],Pfiles[0][1200:1341],label='W='+Wlabel[0])
for num in range(1,len(Pfiles)):
    if num < 8:
        ax.plot(tm[1200:1341],Pfiles[num][1200:1341],label='W='+Wlabel[num])
    elif num > 9:
        ax.plot(tm[1200:1341],Pfiles[num][1200:1341],':',label=item[num][19:-4])
    else:
        ax.plot(tm[1200:1341],Pfiles[num][1200:1341],label=item[num][19:-4])
    #else:
     #   ax.plot(tm[1200:1341],Pfiles[num][1200:1341],label=item[num][19:-4])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
ax.legend(bbox_to_anchor=(1.01, 1.0),ncol=1)
plt.show()

