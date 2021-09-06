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

L=180
delta=-0.15
dt1=0.1
dt2=0.001
dt3=0.1
tint=5
tmid1=6.5
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
Wlabel=['0.0','1e-8','1e-7','1e-6','1e-5','1e-4','1e-3','1e-2','0.1','0.5','1.0']

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



props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textr='\n'.join((
        r'(C1) L = '+str(L)+', $\delta$: '+str(delta)+' -> +'+str(-delta),))

plt.figure(figsize=(12, 4))

ax = plt.subplot(121)

ax.plot(tm,Pfiles[0],label=item[0][25:-10])
for num in range(len(Pfiles)-1,0,-1):
    ax.plot(tm,Pfiles[num],label=item[num][25:-10])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')

ymin=0.17
ymax=0.42
plt.ylim(ymin,ymax)
tm[:1525]
Pfiles[0][:1525]
top=np.amax(Pfiles[0][:1525])
for i in range(1525):
    if top==Pfiles[0][i]:
        tc=tm[i]
plt.axvline(tc,0,(top-ymin)/(ymax-ymin),linestyle=':')
textr2='\n'.join((
        r'$t_{c_1}$',))
plt.text(tc+0.05,ymin+0.005,textr2)
plt.text(5.05,0.9*(ymax-ymin)+ymin,textr)
plt.legend(loc='upper right',fontsize='9')


ax1 = plt.subplot(122)
dy = np.zeros(Pfiles.shape,np.float)
dy[:,0:-1] = np.diff(Pfiles)/np.diff(tm)
dy[:,-1] = (Pfiles[:,-1] - Pfiles[:,-2])/(tm[-1] - tm[-2])

plt.plot(tm[500:900],dy[0][500:900],label=item[0][25:-10])
for num in range(len(Pfiles)-1,0,-1):
    plt.plot(tm[500:900],dy[num][500:900],label=item[num][25:-10])
plt.xlabel('t')
plt.ylabel(r'dl$^\ast$/dt')
#plt.legend(loc='upper right',fontsize='9')
plt.text(7.0,4.2,'(C2)')
plt.show()
print(r'$t_{c_1}$ is ',tc)
