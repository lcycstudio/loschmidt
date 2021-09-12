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

L=18
delta=-0.95
dt1=0.001
dt2=0.05
dt3=0.05
tint=0
tmid1=4
tmid2=8
tmax=9
t1=np.arange(tint,tmid1,dt1)
t2=np.arange(tmid1,tmid2,dt2)
t3=np.arange(tmid2,tmax+dt3,dt3)
tt=np.concatenate((t1,t2), axis=0)
t=np.concatenate((tt,t3), axis=0)
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
#for i in range(len(item2)):
#    f=open(item2[i],'r')
#    AA=f.readlines()
#    f.close()
#    for j in range(len(AA)):
#        Qfiles[a][j]=AA[j]
#    a+=1

"""Plot the Figure """
#tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' and $\mu\in$ [$-W,W$]'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textr='\n'.join((
        r'(B1) L = '+str(L)+', $\delta$: '+str(delta)+' -> +'+str(-delta),))

plt.figure(figsize=(12, 4))

ax = plt.subplot(121)
ymin=-0.2
ymax=4.82
plt.ylim(ymin,ymax)
plt.plot(tm,Pfiles[0],label=item[0][25:-9])
for num in range(len(Pfiles)-1,0,-1):
    plt.plot(tm,Pfiles[num],label=item[num][25:-9])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')


tm[:2000]
Pfiles[0][:2000]
top=np.amax(Pfiles[0][:2000])
for i in range(2000):
    if top==Pfiles[0][i]:
        tc=tm[i]
plt.axvline(tc,0,(top-ymin)/(ymax-ymin),linestyle=':')
textr2='\n'.join((
        r'$t_{c_1}$',))
plt.text(tc+0.1,ymin+(0-ymin)/2,textr2)
plt.text(0.05,0.9*(ymax-ymin)+ymin,textr)
plt.legend(loc='upper right',fontsize='9')


ax1 = plt.subplot(122)
dy = np.zeros(Pfiles.shape,np.float)
dy[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])
plt.plot(tm[1500:1850],dy[0][1500:1850],label=item[0][25:-9])
for num in range(len(Pfiles)-1,0,-1):
    plt.plot(tm[1500:1850],dy[num][1500:1850],label=item[num][25:-9])
plt.xlabel('t')
plt.ylabel(r'dl$^\ast$/dt')
#plt.legend(loc='upper right',fontsize='9')
plt.text(1.5,24,'(B2)')
plt.show()



print(r'$t_{c_1}$ is ',tc)