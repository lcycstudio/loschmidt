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

L=18
delta=-0.95
dt1=0.1
dt2=1e-5
dt3=0.01
tint=1.6
tmid1=1.61
tmid2=1.7
tmax=1.71
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
        r'$\delta$: '+str(delta)+' -> +'+str(-delta),
        r'L = '+str(L)))
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm,Pfiles[0],label=item[0][25:-9])
for num in range(len(Pfiles)-1,0,-1):
        ax.plot(tm,Pfiles[num],':',label=item[num][25:-9])
#for num in range(1,len(Pfiles)):
 #   ax.plot(tm,Pfiles[num],label='W='+Wlabel[num])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
plt.text(1.62,3.3,textr,bbox=props)
ax.legend(loc='upper right')#bbox_to_anchor=(1.0, 1.02),ncol=1)
plt.show()
