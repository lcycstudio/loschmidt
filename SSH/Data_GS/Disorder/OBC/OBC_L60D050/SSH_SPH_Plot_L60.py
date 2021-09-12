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

delta=-0.5
dt1=0.001
dt2=0.01
tint=0
tmid=4
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
Wlabel=['0.0','1e-8','1e-7','1e-6','1e-5','1e-4','1e-3','1e-2','0.1','0.5','1.0']

"""
fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    ax.plot(tm,Pfiles[num],label='W='+Wlabel[num])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
ax.legend(bbox_to_anchor=(1.0, 1.0),ncol=1)
plt.show()


fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    if num <5:
        ax.plot(tm[500:2006],Pfiles[num][500:2006],label='W='+Wlabel[num])
    elif num > 9:
        ax.plot(tm[500:2006],Pfiles[num][500:2006],':',label=item[num][19:-4])
    else:
        ax.plot(tm[500:2006],Pfiles[num][500:2006],label=item[num][19:-4])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
ax.legend(bbox_to_anchor=(1.0, 1.0),ncol=1)
plt.show()


fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm[1000:1400],Pfiles[0][1000:1400],label='W='+Wlabel[0])
for num in range(1,len(Pfiles)):
    if num < 8:
        ax.plot(tm[1000:1400],Pfiles[num][1000:1400],label='W='+Wlabel[num])
    elif num > 9:
        ax.plot(tm[1000:1400],Pfiles[num][1000:1400],':',label=item[num][19:-4])
    else:
        ax.plot(tm[1000:1400],Pfiles[num][1000:1400],label=item[num][19:-4])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
#plt.xlim([1.1,1.4])
ax.legend(bbox_to_anchor=(1.0, 1.02),ncol=1)
plt.show()


fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm[1200:1341],Pfiles[0][1200:1341],label='W='+Wlabel[0])
for num in range(1,len(Pfiles)):
    if num < 8:
        ax.plot(tm[1200:1341],Pfiles[num][1200:1341],label='W='+Wlabel[num])
    elif num > 9:
        ax.plot(tm[1200:1341],Pfiles[num][1200:1341],':',label=item[num][19:-4])
    else:
        ax.plot(tm[1200:1341],Pfiles[num][1200:1341],label=item[num][19:-4])
    #else:
     #   ax.plot(tm[1200:1341],Pfiles[num][1200:1341],label=item[num][19:-4])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
ax.legend(bbox_to_anchor=(1.01, 1.0),ncol=1)
plt.show()
"""


props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textr='\n'.join((
        r'(A1) L = 60'+r', $\delta$: '+str(delta)+' -> +'+str(-delta),))

plt.figure(figsize=(12,4))
ax = plt.subplot(121)


ax.plot(tm,Pfiles[0],label='W='+Wlabel[0])
for num in range(len(Qfiles)-1,-1,-1):
    ax.plot(tm,Qfiles[num],label=item2[num][10:17])
for num in range(1,len(Pfiles)):
    if num > 6:
        plt.plot(tm,Pfiles[num],':',label='W='+Wlabel[num])
    else:
        plt.plot(tm,Pfiles[num],label='W='+Wlabel[num])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')

ymax=1.5
ymin=-0.05
plt.ylim(ymin,ymax)

tm[:4000]
Pfiles[0][:4000]
top=np.amax(Pfiles[0][:4000])
for i in range(4000):
    if top==Pfiles[0][i]:
        tc=tm[i]
plt.axvline(tc,0,(top-ymin)/(ymax-ymin),linestyle=':')
textr2='\n'.join((
        r'$t_{c_1}$',))
plt.text(tc+0.1,ymin+(0-ymin)/2,textr2)
plt.text(0.05,0.9*(ymax-ymin)+ymin,textr)
#plt.text(0.6,0.9*(ymax-ymin)+ymin,textr)#bbox=props)
plt.legend(loc='upper right',fontsize='9')#bbox_to_anchor=(1.0, 1.02),ncol=1)
plt.xlim(-0.5,13.3)


ax1 = plt.subplot(122)

dy = np.zeros(Pfiles.shape,np.float)
dy[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])

dy2 = np.zeros(Qfiles.shape,np.float)
dy2[:,0:-1] = np.diff(Qfiles)/np.diff(tm)
dy2[:,-1] = (Qfiles[:,-1] - Qfiles[:,-2])/(tm[-1] - tm[-2])


props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textr='\n'.join((
        r'$\delta$: '+str(delta)+' -> +'+str(-delta),
        r'L = 60'))
plt.plot(tm[2550:2800],dy[0][2550:2800],label='W='+Wlabel[0])
for num in range(len(Qfiles)-1,-1,-1):
    plt.plot(tm[2550:2800],dy2[num][2550:2800],label=item2[num][10:17])
for num in range(1,len(Pfiles)):
    if num > 6:
        plt.plot(tm[2550:2800],dy[num][2550:2800],':',label='W='+Wlabel[num])
    else:
        plt.plot(tm[2550:2800],dy[num][2550:2800],label='W='+Wlabel[num])
plt.xlabel('t')
plt.ylabel(r'dl$^\ast$/dt')
#plt.text(2.72,7,textr,bbox=props)
#plt.legend(loc='upper right',fontsize='9')#bbox_to_anchor=(1.0, 1.02),ncol=1)
plt.text(2.55,9.5,'(A2)')
plt.show()



plt.show()
print(r'$t_{c_1}$ is ',tc)