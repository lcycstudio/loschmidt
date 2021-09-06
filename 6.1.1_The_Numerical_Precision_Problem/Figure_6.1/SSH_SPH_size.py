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

L=8
delta=-0.5
dt1=0.01
dt2=0.1
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
item3=glob.glob("*.tt")
tm=t
Pfiles=np.zeros((len(item),len(tm)))
Qfiles=np.zeros((len(item2),len(tm)))
Rfiles=np.zeros((len(item3),len(tm)))
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
a=0
for i in range(len(item3)):
    f=open(item3[i],'r')
    AA=f.readlines()
    f.close()
    for j in range(len(AA)):
        Rfiles[a][j]=AA[j]
    a+=1 
    
"""Plot the Figure """
#tiTle='Return Rate for L = '+str(L)+' with $\delta$ = +'+str(delta)+' and $\mu\in$ [$-W,W$]'

fig = plt.figure()
ax = plt.subplot(111)
tiTle1='-->+'+str(-delta)+' and OBC'
tiTleLE2='RR for SSH with L = '+str(L)+', $\delta$ = '+str(delta)
tiTleL=tiTleLE2+tiTle1
for num in range(len(item)):#-2,1,-1):
    if num==0:
        plt.plot(tm,Pfiles[num],label='L='+item[num][33:-4])
    else:
        plt.plot(tm,Pfiles[num],label=item[num][30:-4])
for num in range(len(item2)):
    plt.plot(tm,Qfiles[num],label=item2[num][30:-4])
plt.plot(tm,Rfiles[0],'--',label=item3[0][30:-3])
plt.plot(tm,Rfiles[2],':',label=item3[2][30:-3])
plt.plot(tm,Rfiles[1],':',label=item3[1][30:-3])
plt.xlabel('t')
plt.ylabel(r'l(t)')
plt.ylim(-0.1,1.6)
#plt.title(tiTleL)
plt.legend(bbox_to_anchor=(1.25, 1.02),ncol=1)
iax = plt.axes([-1, 0, 2, 2])
ip = InsetPosition(ax, [0.5, 0.5, 0.45, 0.46]) #posx, posy, width, height
iax.set_axes_locator(ip)
plt.plot(vsf,'o')
plt.ylabel(r'$\lambda_j$')
plt.xlabel('$j$')
#plt.plot(tm,Qfiles[-1],label='W='+item2[-1][28:-8])

plt.show()


top=np.zeros(4)
tcc=np.zeros(4)
for i in range(2,len(item)):
    top[i-2]=np.amax(Pfiles[i][0:200])
    aa=Pfiles[i].tolist()
    bb=int(aa.index(np.amax(aa[0:200])))
    tcc[i-2]=t[bb]

"""        
top[-2]=np.amax(Qfiles[0][0:200])
top[-1]=np.amax(Qfiles[1][0:200])
aa=Qfiles[0].tolist()
bb=int(aa.index(np.amax(aa[0:200])))
tcc[-2]=t[bb]
aa=Qfiles[1].tolist()
bb=int(aa.index(np.amax(aa[0:200])))
tcc[-1]=t[bb]
"""

from scipy.optimize import least_squares

params = np.array([1,1])


def funcinv(x, a, b):
    return b + a*x

def residuals(params, x, data):
    # evaluates function given vector of params [a, b]
    # and return residuals: (observed_data - model_data)
    a, b = params
    func_eval = funcinv(x, a, b)
    return (data - func_eval)

res = least_squares(residuals, params, args=(tcc, top))
print(res)

www=np.arange(tc3[-1],2.669,0.001)
a=0.45666564
b=-0.03879557
y=b+a*www
plt.plot(www,y)

for i in range(len(top2)):
    plt.plot(tc3[i],top2[i],'o')




