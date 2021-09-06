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
dt1=0.1
dt2=0.1
tint=0
tmid=2
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
    for j in range(len(AA)):
        Pfiles[a][j]=AA[j]
    a+=1
    f.close()

"""Plot the Figure """
#tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' and $\mu\in$ [$-W,W$]'



fig, ax = plt.subplots()
#tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' and $\mu\in$ [$-W,W$]'
Llabel=np.asarray([1600,2400,400,600,800,800,1600])#400,600,800,920,1200,2400])
plt.plot(tm,Pfiles[2],'g',label='L='+str(Llabel[2]))
plt.plot(tm,Pfiles[3],'b',label='L='+str(Llabel[3]))
plt.plot(tm,Pfiles[4],'r',label='L='+str(Llabel[4]))
plt.plot(tm,Pfiles[0],'b',label='L='+str(Llabel[0]))
plt.plot(tm,Pfiles[1],'y',label='L='+str(Llabel[1]))
plt.plot(tm,Pfiles[5],'r:',label='L='+str(Llabel[5]))
plt.plot(tm,Pfiles[6],'b:',label='L='+str(Llabel[6]))
plt.plot(tm,Pfiles[7],'g--',label='L='+str(Llabel[5])+r'$^\ast$')
plt.xlim(-0.5,12)
plt.ylim(-0.02,0.7)
plt.legend(loc='upper right')
plt.xlabel('Time t')
plt.ylabel('Return rate l(t)')
plt.show()



delta=-0.5
dt1=0.001
dt2=0.1
tint=0
tmid=2
tmax=10
tt1=np.arange(tint,tmid,dt1)
tt2=np.arange(tmid,tmax+dt2,dt2)
tt=np.concatenate((tt1,tt2), axis=0)   
#dt2=0.05
cwd = os.getcwd() #current working directory
item2=glob.glob("*.txt")
Qfiles=np.zeros((len(item2),len(tt)))
a=0
for i in range(len(item2)):
    f=open(item2[i],'r')
    AA=f.readlines()
    for j in range(len(AA)):
        Qfiles[a][j]=AA[j]
    a+=1
    f.close()

plt.plot(tt,Qfiles[0],'r',label='PBC')
for i in range(1,len(Qfiles)):
    plt.plot(tt,Qfiles[i],label=item2[i][19:-4])
plt.legend(loc='best')
plt.xlabel('Time t')
plt.ylabel('Return rate l(t)')
plt.show()
    
plt.plot(tt,Qfiles[0],'r',label='PBC')
plt.plot(tt,Qfiles[2],'b',label='W=0.1')
#for i in range(1,len(Qfiles)):
 #   plt.plot(tt,Qfiles[i],label=item2[i][19:-4])
plt.legend(loc='best')
plt.xlabel('Time t')
plt.ylabel('Return rate l(t)')
plt.xlim(1.1,1.5)
plt.show()