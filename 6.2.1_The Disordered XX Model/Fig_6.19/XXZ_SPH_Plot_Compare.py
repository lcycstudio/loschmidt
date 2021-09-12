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

delta=-0.5
dt1=0.01
dt2=0.01
tint=0
tmid=3
tmax=10
t1=np.arange(tint,tmid,dt1)
t=np.arange(tint,tmax+dt2/2,dt2)
#t=np.concatenate((t1,t2), axis=0)
#dt2=0.05
cwd = os.getcwd() #current working directory
item=glob.glob("*.txt")
item2=glob.glob("*.dat")
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

tt1=0
tt2=3
tt3=10
dtt=0.1
ttk=np.arange(tt1,tt3+dtt/2,dtt)
item3=glob.glob("*.kk")
qf=np.zeros((len(item3),len(ttk)))
a=0
for i in range(len(item3)):
    f=open(item3[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        qf[a][j]=AA[j]
    a+=1



"""Plot the Figure """

fig=plt.figure(1,figsize=(10,4))
ax1=plt.subplot(121)
plt.plot(tm,-2*np.log(Qfiles[0])/200,label='L=100(OBC)')
plt.plot(tm,-2*np.log(Qfiles[1])/200,label='L=100(PBC)')
plt.xlabel('t')
plt.ylabel(r'l(t)')
plt.text(0,0.75,'(A)')
plt.legend(loc='upper right')#bbox_to_anchor=(1.01, 1.00),ncol=1)

RR1=-2*np.log(Pfiles[1])/800
RR2=-2*np.log(Pfiles[2])/800
RR3=-2*np.log(qf[0])/1600
ax2=plt.subplot(122)
plt.plot(tm,RR1,label='L=800(OBC)')
plt.plot(tm,RR2,label='L=800(PBC)')
plt.plot(tm,Pfiles[0],label='Analytic(PBC)')
plt.plot(ttk,RR3,'--',label='1600(PBC)')
plt.xlabel('t')
plt.text(0,1.4,'(B)')
plt.legend(loc='upper right')#bbox_to_anchor=(1.01, 1.00),ncol=1)
plt.show()


plt.plot(tm[200:250],RR1[200:250],label='L=800')
plt.plot(tm[200:250],Pfiles[0][200:250],label='Analytic')
plt.legend(loc='best')