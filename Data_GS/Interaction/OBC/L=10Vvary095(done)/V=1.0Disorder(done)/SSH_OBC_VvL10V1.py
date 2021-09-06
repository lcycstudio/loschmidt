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
V=1.0
delta=-0.95
dt1=0.01
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
    for j in range(len(tm)):
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

#Pfiles=-2*np.log(Pfiles)/L
"""Plot the Figure """
wb=['1e-4','1e-3','1e-2','5e-2','1e-1']
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textr='\n'.join((
        r'(C) $\Delta$: '+str(0)+' -> '+str(V),))
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm,Qfiles[0],label=item2[0][:-4])
for num in range(len(Pfiles)):
    ax.plot(tm,Pfiles[num],label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'l$^\ast$(t)')
plt.ylim(-0.05,1.35)
plt.text(0.3,1.2,textr)
ax.legend(loc='upper right',fontsize='9')#bbox_to_anchor=(1.0, 1.02),ncol=1)
plt.show()
