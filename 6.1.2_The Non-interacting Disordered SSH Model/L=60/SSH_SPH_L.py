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

L=60
delta1="0.95"
delta2="0.50"
delta3="0.15"
dt1=0.01
dt2=0.01
tint=0
tmid=2
tmax=10
t1=np.arange(tint,tmid,dt1)
t2=np.arange(tmid,tmax+dt2,dt2)
t=np.concatenate((t1,t2), axis=0)
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

fig = plt.figure()
ax = plt.subplot(111)
#tiTle1='-->+'+str(-delta)+' and OBC'
#tiTleLE2='RR for SSH with L = '+str(L)+', $\delta$ = '+str(delta)
#tiTleL=tiTleLE2+tiTle1
plt.plot(tm,Pfiles[2],color='dodgerblue',label=r'$\delta$='+str(delta1)+'('+item[2][4:7]+')')
plt.plot(tm,Pfiles[1],color='orange',label=r'$\delta$='+str(delta2)+'('+item[1][4:7]+')')
plt.plot(tm,Pfiles[0],color='green',label=r'$\delta$='+str(delta3)+'('+item[0][4:7]+')')
plt.plot(tm,Pfiles[5],color='red',linestyle='dashed',label=r'|$\delta$|='+str(delta1)+'('+item[5][4:7]+')')
plt.plot(tm,Pfiles[4],color='m',linestyle='dashed',label=r'|$\delta$|='+str(delta2)+'('+item[4][4:7]+')')
plt.plot(tm,Pfiles[3],color='brown',linestyle='dashed',label=r'|$\delta$|='+str(delta3)+'('+item[3][4:7]+')')

#for num in range(len(item2)):
#    plt.plot(tm,Qfiles[num],label=item2[num][30:-4])
plt.xlabel('t')
plt.ylabel(r'l(t)')
#plt.text(0.1,3.5,'(D)')
#plt.ylim(-0.1,1.6)
#plt.title(tiTleL)
plt.legend(loc='best')#bbox_to_anchor=(1.25, 1.02),ncol=1)
#iax = plt.axes([-1, 0, 2, 2])
#ip = InsetPosition(ax, [0.5, 0.5, 0.45, 0.46]) #posx, posy, width, height
#iax.set_axes_locator(ip)
#plt.plot(vsf,'o')
#plt.ylabel(r'$\lambda_j$')
#plt.xlabel('$j$')
#plt.plot(tm,Qfiles[-1],label='W='+item2[-1][28:-8])

plt.show()



