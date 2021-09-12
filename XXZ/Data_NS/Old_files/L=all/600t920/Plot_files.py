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
dt=0.05
#dt2=0.05
cwd = os.getcwd() #current working directory
item=glob.glob("*.dat")
tn=np.arange(0,tmax,dt)
Pfiles=np.zeros((len(item),len(tn)))
a=0
for i in range(len(item)):
    f=open(item[i],'r')
    AA=f.readlines()
    for j in range(len(AA)):
        Pfiles[a][j]=AA[j]
    a+=1
Llabel=np.asarray([600,800,920])
plt.plot(tn,Pfiles[0],'b--',label='L='+str(Llabel[0]))
plt.plot(tn,Pfiles[1],'r',label='L='+str(Llabel[1]))
plt.plot(tn,Pfiles[2],'y:',label='L='+str(Llabel[2]))
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
#fig, ax = plt.subplots()
Llabel=np.asarray([600,800,920])
plt.plot(tm,Qfiles[0],'b--',label='L='+str(Llabel[0]))
plt.plot(tm,Qfiles[1],'r',label='L='+str(Llabel[1]))
plt.plot(tm,Qfiles[2],'y:',label='L='+str(Llabel[2]))
plt.legend(loc='upper right')
plt.xlabel('t')
plt.ylabel('l(t)')
plt.show()


plt.plot(tm,Qfiles[0],'b--',label='L='+str(Llabel[0]))
plt.plot(tm,Qfiles[1],'r',label='L='+str(Llabel[1]))
plt.plot(tm,Qfiles[2],'y:',label='L='+str(Llabel[2]))
plt.legend(loc='upper right')
plt.xlim([0.75,0.9])
plt.ylim([1,1.65])
plt.xlabel('t')
plt.ylabel('l(t)')
plt.show()

"""
iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax, [0.4, 0.05, 0.4, 0.4]) #posx, posy, width, height
iax.set_axes_locator(ip)
ss=Qfiles[:,1520:1800]
for i in ss:
    plt.plot(i)
plt.ylim([1.2,1.8])
plt.xticks([])
plt.yticks([])
plt.show()
"""
