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

L=800
delta=0.3
tmax=10
dt=0.01
#dt2=0.05
cwd = os.getcwd() #current working directory
item1=glob.glob("*.dat")
tn=np.arange(0,tmax,dt)
Pfiles=np.zeros((len(item1),len(tn)))
a=0
for i in range(len(item1)):
    f=open(item1[i],'r')
    AA=f.readlines()
    for j in range(len(AA)):
        Pfiles[a][j]=AA[j]
    a+=1
plt.plot(tn,Pfiles[2],'r',label='OBC, $\delta$=+'+str(delta)+'-->-'+str(delta))
plt.plot(tn,Pfiles[0],'b-',label='OBC, $\delta$=-'+str(delta)+'-->+'+str(delta))
plt.plot(tn,Pfiles[3],'g:',label='PBC, $\delta$=+'+str(delta)+'-->-'+str(delta))
plt.plot(tn,Pfiles[1],'y:',label='PBC, $\delta$=-'+str(delta)+'-->+'+str(delta))
plt.legend(loc='upper right')
plt.xlabel('t')
plt.ylabel('l(t)')
plt.show()


tmax=2
dt=0.0005
item=glob.glob("*.txt")
tm=np.arange(0,tmax,dt)
Qfiles=np.zeros((len(item),len(tm)))
a=0
for i in range(len(item)):
    f=open(item[i],'r')
    BB=f.readlines()
    for j in range(len(BB)):
        Qfiles[a][j]=BB[j]
    a+=1

"""Plot the Figure """
fig, ax = plt.subplots()
plt.plot(tm,Qfiles[2],'r',label='OBC, $\delta$=+'+str(delta)+'-->-'+str(delta))
plt.plot(tm,Qfiles[0],'b-',label='OBC, $\delta$=-'+str(delta)+'-->+'+str(delta))
plt.plot(tm,Qfiles[3],'g:',label='PBC, $\delta$=+'+str(delta)+'-->-'+str(delta))
plt.plot(tm,Qfiles[1],'y:',label='PBC, $\delta$=-'+str(delta)+'-->+'+str(delta))
plt.legend(loc='upper right')
plt.ylim([-0.1,1.8])
plt.xlabel('t')
plt.ylabel('l(t)')

iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax, [0.4, 0.05, 0.4, 0.4]) #posx, posy, width, height
iax.set_axes_locator(ip)
ss=Qfiles[:,1520:1800]
plt.plot(ss[2],'r:')
plt.plot(ss[0],'b:')
plt.plot(ss[3],'g')
plt.plot(ss[1],'y:')
plt.ylim([1.2,1.8])
plt.xticks([])
plt.yticks([])
plt.show()

