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

L=14
delta=-0.5
W=0.5
V=1.0
tint=0
tmax=50
dt=0.25
tm=np.arange(tint,tmax+dt/2,dt)

cwd = os.getcwd() #current working directory
item=glob.glob("*.txt")
Pfiles=np.zeros((len(item),len(tm)))
a=0
for i in range(len(item)):
    f=open(item[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        Pfiles[a][j]=AA[j]
    a+=1


"""Plot the Figure """
#Wlabel=['0.0','5e-8','5e-7','5e-6','5e-5']
fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    if num == 3:
        ax.plot(tm,Pfiles[num],'--',label=item[num][23:-4])
    else:
        ax.plot(tm,Pfiles[num],label=item[num][23:-4])
plt.xlabel('Time t')
plt.ylabel('Return rate l(t)')
ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1)
plt.show()


fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    if num == 3:
        ax.plot(tm,Pfiles[num],'--',label=item[num][23:-4])
    else:
        ax.plot(tm,Pfiles[num],label=item[num][23:-4])
plt.xlim([0,10])
plt.xlabel('Time t')
plt.ylabel('Return rate l(t)')
ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1)
plt.show()


fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    if num == 3:
        ax.plot(tm[40:121],Pfiles[num][40:121],'--',label=item[num][23:-4])
    else:
        ax.plot(tm[40:121],Pfiles[num][40:121],label=item[num][23:-4])
plt.xlabel('Time t')
plt.ylabel('Return rate l(t)')
ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1)
plt.show()

