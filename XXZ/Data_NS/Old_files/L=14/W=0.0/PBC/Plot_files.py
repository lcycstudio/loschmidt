# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:34:20 2017

@author: LewisCYC
"""
import os             
import glob
import numpy as np
import matplotlib.pyplot as plt

L=14
dt=0.01
delta=0.0
W=0.0
cwd = os.getcwd() #current working directory
item=glob.glob("*.dat")
tm=np.arange(0,20,dt)

"""Plot the Figure """
item=glob.glob("*.dat")
LE=np.zeros((len(item),len(tm)))
#LE=np.zeros((len(item),len(tm)))
for i in range(len(item)):
    f = open(item[0], "r")
    lines = f.readlines()
    for k in range (0,len(tm)):
        LE[i][k]=float(lines[k])
f.close()

RR=-2*np.log(LE)/L
tiTle='Return Rate for L = 14 with $\delta$ = '+str(delta)+' and W = '+str(W)
#plt.plot(tm,LE[1],'r',label='$\mathcal{L}(t)=|det(1-C+Ce^{-iHt})|^2$ (Single-particle H)')
#plt.plot(tm,LE[0],'b--',label='$\mathcal{L}(t)=|<exp(-iHt)>|^2$ (Many-particle H)')
plt.plot(tm,LE[1],'r',label='SPH Approach')
plt.plot(tm,LE[0],'b--',label='MPH Approach')
plt.legend(loc='upper right')
plt.ylabel('$\mathcal{L}(t)$')
plt.xlabel('t')
plt.show()

#plt.plot(tm,RR[1],'r',label='$l(t)$ for SPH Case')
#plt.plot(tm,RR[0],'b--',label='$l(t)$ for MPH Case')
plt.plot(tm,RR[1],'r',label='SPH Approach')
plt.plot(tm,RR[0],'b--',label='MPH Approach')
plt.legend(loc='upper right');
plt.ylabel('l(t)')
plt.xlabel('t')
plt.show()
#plt.ylim((-0.05,2.6))
#plt.title('Return Rate for L = '+str(L)+' with $\delta$ = '+str(delta)+' and W = '+str(W),fontsize=12)
