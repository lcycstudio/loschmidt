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
dt1=0.001
dt2=0.1
tint=0
tmid=2
tmid2=4
tmax=10
t1=np.arange(tint,tmid,dt1)
t2=np.arange(tmid,tmax+dt2,dt2)
t=np.concatenate((t1,t2), axis=0)
tn1=np.arange(tint,tmid2,dt1)
tn2=np.arange(tmid2,tmax+dt2,dt2)
tn=np.concatenate((tn1,tn2), axis=0)
#dt2=0.05
cwd = os.getcwd() #current working directory
item=glob.glob("*.txt")
item2=glob.glob("*.dat")
tm=t
Pfiles=np.zeros((len(item),len(tm)))
a=0
for i in range(len(item)):
    f=open(item[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        Pfiles[a][j]=AA[j]
    a+=1

qf=np.zeros((len(item2),len(tn)))
a=0
for i in range(len(item2)):
    f=open(item2[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        qf[a][j]=AA[j]
    a+=1
    

for cc in range(len(item)):
    plt.plot(tm,Pfiles[cc])
plt.show()



"""Plot the Figure """
#tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' and $\mu\in$ [$-W,W$]'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
a=0.5221
b=-0.1229
www=np.arange(1.245,2.669,0.001)
y=b+a*www
Wlabel=['5e-08','5e-07','5e-06','5e-05','5e-04','5e-03','1e-02','5e-02','5e-14']
fig = plt.figure()
ax = plt.subplot(111)
plt.plot(www,y,'k')
plt.plot(tn,qf[0],'r--',label='L=60(clean)')

ax.plot(tm,Pfiles[-1],label='W='+Wlabel[-1])
for num in range(0,len(Pfiles)-1):
    if num < 8:
        ax.plot(tm,Pfiles[num],label='W='+Wlabel[num])
    elif num >= 8:
        ax.plot(tm,Pfiles[num],':',label=item[num][19:-4])
    else:
        ax.plot(tm,Pfiles[num],label=item[num][19:-4])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
plt.axvline(x=2.669,linestyle=':')
plt.text(2.75,-0.02,r'$t_{c_1}$')
ax.legend(bbox_to_anchor=(1.0, 1.02),ncol=1,fontsize='9')
textr='\n'.join((
        'hypothetical line of convergence:',
        r'$y(t_{c^\ast})$='+str(round(a,4))+r'*$t_{c^\ast}$'+str(round(b,4)),))
plt.text(3.5,1.25,textr,bbox=props)

plt.ylim(-0.05,1.5)
plt.text(0.05,0.9*1.5,'(A)')
iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax, [0.4, 0.35, 0.4, 0.4]) #posx, posy, width, height
iax.set_axes_locator(ip)
plt.plot(tm[1000:1400],Pfiles[-1][1000:1400],label='W='+Wlabel[-1])
for num in range(0,len(Pfiles)-1):
    if num < 8:
        plt.plot(tm[1000:1400],Pfiles[num][1000:1400])#,label='W='+Wlabel[num])
    elif num >= 8:
        plt.plot(tm[1000:1400],Pfiles[num][1000:1400],':')#,':',label=item[num][19:-4])
    else:
        plt.plot(tm[1000:1400],Pfiles[num][1000:1400])#,label=item[num][19:-4])
plt.xticks([])
plt.yticks([])
plt.show()

"""
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
ax.legend(bbox_to_anchor=(1.01, 1.0),ncol=1)
plt.show()
"""

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(tm[1000:1400],Pfiles[-1][1000:1400],label='W='+Wlabel[-1])
for num in range(0,len(Pfiles)-1):
    if num < 8:
        ax.plot(tm[1000:1400],Pfiles[num][1000:1400],label='W='+Wlabel[num])
    elif num >= 8:
        ax.plot(tm[1000:1400],Pfiles[num][1000:1400],':',label=item[num][19:-4])
    else:
        ax.plot(tm[1000:1400],Pfiles[num][1000:1400],label=item[num][19:-4])
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
plt.xlim([1.1,1.4])
ax.legend(bbox_to_anchor=(1.0, 1.02),ncol=1,fontsize='9')
ymin=0.375
ymax=0.59
plt.ylim(ymin,ymax)
plt.text(1.12,0.9*(ymax-ymin)+ymin,'(B)')
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

plt.plot(tn,qf[0],'r--',label='L=60(clean)')
plt.axhline(y=np.amax(qf[0]),linestyle=':')
plt.axhline(y=1.2705849,linestyle=':')
plt.show()

print('tc is',tc)




