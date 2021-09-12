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

delta=0.3
#W=1.0
tmid=2
tmax=10
dt=0.01
#dt2=0.05
cwd = os.getcwd() #current working directory
item=glob.glob("*.txt")
tm=np.arange(0,tmax,dt)
Pfiles=np.zeros((len(item),len(tm)))
a=0
for i in range(len(item)):
    f=open(item[i],'r')
    AA=f.readlines()
    for j in range(len(AA)):
        Pfiles[a][j]=AA[j]
    a+=1

"""Plot the Figure """
#tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' and $\mu\in$ [$-W,W$]'



fig, ax = plt.subplots()
#tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' and $\mu\in$ [$-W,W$]'
Llabel=np.asarray([50,100,200,400,600,800,920,1200,2400])

for num in range(len(Pfiles)):
    plt.plot(tm,Pfiles[num],label='L='+str(Llabel[num]))
plt.legend(loc='upper right')
plt.xlabel('t')
plt.ylabel('l(t)')
plt.show()

for num in range(len(Pfiles)):
    plt.plot(tm,Pfiles[num],label='L='+str(Llabel[num]))
plt.legend(loc='upper right')
plt.xlim([0,0.5])
plt.xlabel('t')
plt.ylabel('l(t)')
plt.show()
