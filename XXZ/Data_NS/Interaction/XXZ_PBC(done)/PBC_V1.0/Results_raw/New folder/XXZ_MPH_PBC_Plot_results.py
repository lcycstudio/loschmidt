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
V=1.0
W=0.0001
tint=0
tmax=50
dt=0.005    
openbc=0
if openbc==1:
    bc='OBC'
else:
    bc='PBC'
cwd = os.getcwd() #current working directory
item=glob.glob("*.txt")
tm=np.arange(0,tmax+dt/2,dt)
Pfiles=np.zeros((len(item),len(tm)))
a=0
for i in range(len(item)):
    f=open(item[i],'r')
    AA=f.readlines()
    for j in range(len(AA)):
        Pfiles[a][j]=AA[j]
    a+=1
    f.close()

"""Plot the Figure """
wb=['0.0','1e-4','1e-3','0.01','0.06','0.1','1.0','3.5','6.0']
tiTle='Return Rate for XXZ model with '+bc
fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    ax.plot(tm,Pfiles[num],label='W='+wb[num])
plt.xlabel('t')
plt.ylabel(r'l$^\ast$(t)')
plt.ylim(-0.05,1.0)
plt.xlim(-2,60)
plt.text(0.3,0.9,'(A) V = 1.0')
ax.legend(loc='upper right',fontsize='9')
#plt.title(tiTle)
plt.show()

"""
fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    if num == 3:
        ax.plot(tm[4000:8001],Pfiles[num][4000:8001],'--',label=item[num][13:-4])
    else:
        ax.plot(tm[4000:8001],Pfiles[num][4000:8001],label=item[num][13:-4])
plt.xlabel('t')
plt.ylabel('Return rate l(t)')
ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1,fontsize='9')
#plt.title(tiTle)
plt.show()

fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    ax.plot(tm[500:2006],Pfiles[num][500:2006],label=item[num][19:-4])
plt.xlabel('t')
plt.ylabel('l(t)')
ax.legend(bbox_to_anchor=(1.01, 1.0),ncol=1)
plt.show()
"""


