# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:57:26 2018

@author: LewisCYC
"""

import os             
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
#from scipy.optimize import curve_fit

#from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
L=800
delta=-0.5
dt1=0.001
dt2=0.1
tint=0
tmid=2
tmax=10
t1=np.arange(tint,tmid,dt1)
t2=np.arange(tmid,tmax+dt2,dt2)
tm=np.concatenate((t1,t2), axis=0)
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

item2=glob.glob("*.dat")
Qfiles=np.zeros((len(item2),len(tm)))
a=0
for i in range(len(item2)):
    f=open(item2[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        Qfiles[a][j]=AA[j]
    a+=1
    

""" Smoothing out W=0.0 """
maxIndex=Pfiles[0].tolist().index(np.amax(Pfiles[0]))
tn=np.arange(maxIndex,2000,30)
tIndex=np.concatenate((tn,np.array([2000])), axis=0)
x=np.zeros(len(tIndex))
y=np.zeros(len(x))
for jj in range(len(x)):
    y[jj]=Pfiles[0][tIndex[jj]]
    x[jj]=tm[tIndex[jj]]
z=np.polyfit(x,y,2)
p=np.poly1d(z)
dt3=len(Pfiles[0][maxIndex:2000])
xp=np.linspace(x[0],x[-1],dt3)
yp=p(xp)
fig = plt.figure()
ax = plt.subplot(111)
Wlabel=['0.0','5e-7','5e-6','5e-5']
ax.plot(xp,p(xp),'k:',label='Polyfit')
Pfiles[0][maxIndex:2000]=yp
tm0=tm
tm0[maxIndex:2000]=xp


"""Plots of the Data """
Wlabel=['0.0','5e-8','5e-7','5e-6','5e-5','5e-4','5e-3','5e-2']
fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    if num <5:
        if num == 0:
            ax.plot(tm0,Pfiles[num],label='W='+Wlabel[num])
        else:
            ax.plot(tm,Pfiles[num],label='W='+Wlabel[num])
    elif num > 9:
        ax.plot(tm,Pfiles[num],':',label='W='+Wlabel[num])
    else:
        ax.plot(tm,Pfiles[num],label='W='+Wlabel[num])
for kk in range(len(item2)):
    ax.plot(tm,Qfiles[kk],'--',label=item2[kk][-11:-4])
plt.xlim(1.0,1.6)
plt.ylim(0.25,0.6)
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1)
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
plt.xlabel('Time t')
plt.ylabel('Return rate l(t)')
ax.legend(bbox_to_anchor=(1.01, 1.0),ncol=1)
plt.show()
"""

""" Plot peak points """
Peaks=np.zeros(len(Pfiles))
Ptime=np.zeros(len(Pfiles))
leftPeaks=np.zeros(len(Pfiles))
leftPtime=np.zeros(len(Pfiles))
for i in range(len(Pfiles)):
    maxIndex=Pfiles[i].tolist().index(np.amax(Pfiles[i]))
    leftPeaks[i]=Pfiles[i][maxIndex-5]
    leftPtime[i]=tm[maxIndex-5]
    Peaks[i]=Pfiles[i][maxIndex]
    Ptime[i]=tm[maxIndex]
fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Peaks)):
    if num<5:
        ax.plot(Ptime[num],Peaks[num],'o',label='W='+Wlabel[num])
    else:
        ax.plot(Ptime[num],Peaks[num],'o',label=item[num][19:-4])
plt.title('Peaks of the cusps')
ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1)
plt.show()


""" Plot polynomial fitting of the peaks """
fig = plt.figure()
ax = plt.subplot(111)
x1=Ptime
y1=Peaks
z1=np.polyfit(x1,y1,1)
p1=np.poly1d(z1)
xp1=np.linspace(x1[0],x1[-1],1000)
yp1=p1(x1)#xp1*z1[0]+z1[1]
for num in range(len(x1)):
    if num <5:
        ax.plot(x1[num],y1[num],'o',label='W='+Wlabel[num])
    else:
        ax.plot(x1[num],y1[num],'o',label='W='+Wlabel[num])

plt.plot(xp1,p1(xp1),label='Polyfit')
plt.legend(loc='best')
plt.ylabel('Peaks of l(t)')
plt.xlabel('Time t')
#ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1)
plt.show()



    
""" Plot peak points """
Qeaks=np.zeros(len(Qfiles))
Qtime=np.zeros(len(Qfiles))
leftQeaks=np.zeros(len(Qfiles))
leftQtime=np.zeros(len(Qfiles))
for i in range(len(Qfiles)):
    maxIndex=Qfiles[i].tolist().index(np.amax(Qfiles[i]))
    leftPeaks[i]=Qfiles[i][maxIndex-5]
    leftPtime[i]=tm[maxIndex-5]
    Qeaks[i]=Qfiles[i][maxIndex]
    Qtime[i]=tm[maxIndex]
fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Qeaks)):
    if num<5:
        ax.plot(Qtime[num],Qeaks[num],'o',label='W='+Wlabel[num])
    else:
        ax.plot(Qtime[num],Qeaks[num],'o',label=item[num][19:-4])
plt.title('Peaks of the cusps')
ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1)
plt.show()


y2=np.zeros(2)
y2[0]=np.amax(Qfiles[1])
y2[1]=np.amax(Qfiles[0])
y3=np.delete(y1,0)
yy=np.concatenate((y2,y3),axis=0)
x2=np.zeros(2)
x2[0]=1.348
x2[1]=1.342
x3=np.delete(x1,0)
xx=np.concatenate([x2,x3],axis=0)
z2=np.polyfit(xx,yy,1)
p2=np.poly1d(z2)
xp2=np.linspace(xx[0],xx[-1],1000)
yp2=p2(xx)

fig = plt.figure()
ax = plt.subplot(111)
x1=Ptime
y1=Peaks
z1=np.polyfit(x1,y1,1)
p1=np.poly1d(z1)
xp1=np.linspace(x1[0],x1[-1],1000)
yp1=p1(x1)#xp1*z1[0]+z1[1]
for num in range(len(x1)):
    if num <5:
        ax.plot(x1[num],y1[num],'o',label='W='+Wlabel[num])
    else:
        ax.plot(x1[num],y1[num],'o',label='W='+Wlabel[num])
for kk in range(len(Qeaks)):
    ax.plot(Qtime[kk],Qeaks[kk],'o',label=item2[kk][-11:-4])
plt.plot(xp1,p1(xp1),label='Old Polyfit')
plt.plot(xp2,p2(xp2),label='New Polyfit')
plt.legend(loc='best')
plt.xlabel('Time t')
plt.ylabel('Peaks of l(t)')
plt.xlim([1.23,1.41])
plt.xticks([x1[0],x1[-1]])
plt.axvline(x1[0],color=colors['grey'],linestyle='--')
#plt.axvline(x=np.pi/2,color=colors['grey'],linestyle='--')

#ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1)
plt.show()




""" Step difference """
dy = np.diff(Peaks[::-1])
dt = np.diff(Ptime[::-1])
dt2=np.sum(dt[:-1],axis=0)/(len(dt)-1)
hgtd=(Ptime[0]-Ptime[1])/dt2


""" Plot data fileName"""
"""
fileW=np.asarray([float(x) for x in fileName])
xw=Ptime
yw=fileW
zw=np.polyfit(xw,yw,3)
pw=np.poly1d(zw)
xpw=np.linspace(xw[0],xw[-1],1000)
ypw=xp*zw[0]+zw[1]
params = np.array([1,1])
def funcinv(x, a, b, c):
    return a * np.exp(-b * x) + c
popt,pcov = curve_fit(funcinv, xw, yw)
plt.plot(xw, funcinv(xw, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
for num in range(len(xw)):
    plt.plot(xw[num],yw[num],'*')
plt.plot(xpw,pw(xpw),'k:')
plt.title('Points on the line')
plt.show()
"""



dy = np.diff(Peaks[::-1])
dt = np.diff(Ptime[::-1])
tbar = []
for i in range(1,len(Ptime)):
    tbar.append('['+str(Ptime[i])+','+str(Ptime[i])+']')
    print('['+str(Ptime[i-1])+','+str(Ptime[i])+']')





ss=np.zeros(10)
ss[0]=1.293
for i in range(len(ss)-1):
    ss[i+1]=ss[i]+0.008
ss2=ss[1:-2]
ss2=ss2[::-1]
indt=np.zeros(7)
aa=np.zeros(15)
a1=0
for i in range(len(xp1)):
    for j in ss2:
        if np.abs(xp1[i]-j)<=0.0001:
            print(i)
            aa[a1]=i
            a1=a1+1
for i in range(0,len(aa),2):
    if aa[i]!=0:
        j=int(i/2)
        indt[j+1]=aa[i]
pp=np.zeros(len(indt))
for i in range(len(indt)):
    pp[i]=p1(xp1)[int(indt[i])]
plt.plot(x1,yp1,'o')
ss2[0]=1.348
WWlabel=['5e-15','5e-14','5e-13','5e-12','5e-11','5e-10','5e-9',]
plt.plot(ss2[0],pp[0],'*',label='W=0.0')
for i in range(len(pp)):
    plt.plot(ss2[i],pp[i],'o',label='W='+WWlabel[i])
plt.plot(xp1,p1(xp1),label='Polyfit')
plt.xticks([x1[0],x1[-1]])
plt.axvline(x1[0],color=colors['grey'],linestyle='--')
plt.legend(bbox_to_anchor=(0.79, 0.51, 0.5, 0.5))#loc='best')
plt.xlabel('Time t')
plt.ylabel(r'Peaks of l$^\ast$(t)')
plt.text(x1[1],yp1[2],'W=5e-8')
plt.text(x1[2],yp1[3],'W=5e-7')


from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
fig, ax = plt.subplots()

#fig = plt.figure()
#ax = plt.subplot(111)
Qtime[1]=1.347
x1=Ptime
y1=Peaks
z1=np.polyfit(x1,y1,1)
p1=np.poly1d(z1)
xp1=np.linspace(x1[0],x1[-1],1000)
yp1=p1(x1)#xp1*z1[0]+z1[1]
for num in range(len(x1)):
    if num <5:
        ax.plot(x1[num],y1[num],'o',label='W='+Wlabel[num])
    else:
        ax.plot(x1[num],y1[num],'o')#,label='W='+Wlabel[num])
plt.plot(ss2[1],pp[1],'*',label='W='+WWlabel[1]+'(fit)')
plt.plot(ss2[0],pp[0],'*',label='W='+WWlabel[0]+'(fit)')
for kk in range(len(Qeaks)):
    ax.plot(Qtime[kk],Qeaks[kk],'*',label=item2[kk][-11:-4]+'(data)')
plt.plot(xp1,p1(xp1),color=colors['dodgerblue'],label='Polyfit')
#plt.plot(xp2,p2(xp2),'orange',label='New Polyfit')
plt.legend(bbox_to_anchor=(0.87, 0.51, 0.5, 0.5))#loc='best')
plt.xlabel('Time t')
plt.ylabel(r'Peaks of l$^\ast$(t)')
#plt.xlim([1.23,1.43])
plt.xticks([x1[-1],x1[1],ss2[1]])
plt.axvline(x1[0],color=colors['grey'],linestyle='--')
plt.axvline(ss2[1],color=colors['grey'],linestyle='--')
plt.axvline(Qtime[0],color=colors['grey'],linestyle='--')
#plt.axvline(x=np.pi/2,color=colors['grey'],linestyle='--')

iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax, [0.08, 0.55, 0.4, 0.4]) #posx, posy, width, height
iax.set_axes_locator(ip)
plt.plot(xp1,p1(xp1))
#plt.plot(xp2,p2(xp2))
plt.xlim([1.34,1.349])
plt.ylim([0.575,0.585])
plt.xticks([Qtime[0],Qtime[1]],fontsize=8)
plt.axvline(x1[0],color=colors['grey'],linestyle='--')
plt.axvline(ss2[1],color=colors['grey'],linestyle='--')
plt.axvline(Qtime[0],color=colors['grey'],linestyle='--')
plt.axvline(Qtime[1],color=colors['grey'],linestyle='--')
plt.yticks([])
plt.plot(Qtime[0],Qeaks[0],'*',color=colors['steelblue'])
plt.plot(Qtime[1],Qeaks[1],'*',color=colors['orange'])#,label='W='+WWlabel[0]+'(old)')
plt.plot(ss2[1],pp[1],'y*')#,label='W='+WWlabel[0]+'(new)')
plt.plot(ss2[0],pp[0],'*',color=colors['skyblue'])#,label='W='+WWlabel[1]+'(new)')
plt.legend(loc='bottom')
plt.show()



"""Plots of the Data """
Wlabel=['0.0','5e-8','5e-7','5e-6','5e-5','5e-4','5e-3','5e-2']
fig = plt.figure()
ax = plt.subplot(111)
for num in range(len(Pfiles)):
    if num <5:
        if num == 0:
            ax.plot(tm0,Pfiles[num],label='W='+Wlabel[num])
        else:
            ax.plot(tm,Pfiles[num],label='W='+Wlabel[num])
    elif num > 9:
        ax.plot(tm,Pfiles[num],':',label='W='+Wlabel[num])
    else:
        ax.plot(tm,Pfiles[num],label='W='+Wlabel[num])
for kk in range(len(item2)):
    ax.plot(tm,Qfiles[kk],'--',label=item2[kk][-11:-4])
plt.xlim(1.0,1.6)
plt.ylim(0.25,0.6)
plt.xlabel('t')
plt.ylabel(r'l$^{\ast}$(t)')
ax.legend(bbox_to_anchor=(1.01, 1.00),ncol=1)
plt.show()


