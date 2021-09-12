# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 01:36:16 2018

@author: LewisCYC
"""

import os             
import glob
import numpy as np
import matplotlib.pyplot as plt


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
    
rc=np.zeros(len(item))
tc=np.zeros(len(item))
for i in range(len(item)):
    rc[i]=np.amax(Pfiles[i])
    for j in range(len(tm)):
        if rc[i]==Pfiles[i][j]:
            tc[i]=tm[j]
            
Wlabel=['5e-8','5e-7','5e-6','5e-5','5e-4','5e-3','5e-2','0.1','0.25','0.5','1.0']

ww=np.zeros(len(item))
for i in range(len(ww)):
    ww[i]=float(item[i][21:-4])
    
plt.plot(ww[0],tc[0],'o',label=item[0][19:-4])
plt.plot(ww[-1],tc[-1],'o',label=item[-1][19:-4])
plt.plot(ww[-2],tc[-2],'o',label=item[-2][19:-4])
for i in range(1,len(ww)-2):
    if i==11:
        plt.plot(ww[i],tc[i],'o',label=item[i][19:-4])
    else:
        plt.plot(ww[i],tc[i],'o')#,label=item[i][19:-4])
plt.legend(loc='best')
plt.yticks([tc[0],tc[-2],tc[11]])
plt.xticks([0.0,0.1,0.2,0.4,0.6,0.8,1.0])
plt.ylabel(r'$t_{c^\ast}$')
plt.xlabel('W')
plt.show()

tc2=np.zeros(len(tc)-4)
tc2[0]=tc[0]
tc2[1]=tc[-2]
tc2[2:]=tc[1:-5]
ww2=np.zeros(len(ww)-4)
ww2[0]=ww[-1]
ww2[1]=ww[-2]
ww2[2:]=ww[1:-5]
rc2=np.zeros(len(rc)-4)
rc2[0]=rc[-1]
rc2[1]=rc[-2]
rc2[2:]=rc[1:-5]


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

res = least_squares(residuals, params, args=(np.log(ww2), np.log(tc2)))
res2 = least_squares(residuals, params, args=(tc2, rc2))
print(res)
print(res2)

L=800
delta=-0.5
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
www=np.arange(np.log(ww2[0]),np.log(ww2[-1]),0.1)
a=-0.00261123
b=0.2134359
y=b+a*www
plt.plot(www,y)
textr='\n'.join((
        r'$\delta$: '+str(delta)+' -> +'+str(-delta),
        r'L = '+str(L),
        r'$ln(t_{c^\ast})$='+str(round(a,4))+r'*$ln(W)$+'+str(round(b,4))))
ww3=['5e-15','5e-14','5e-08','5e-07','5e-06','5e-05','5e-04','5e-03','1e-02','5e-02','0.1','0.25','0.5','1.0']
for i in range(len(ww2)):
    plt.plot(np.log(ww2[i]),np.log(tc2[i]),'o',label='W='+str(ww3[i]))
plt.text(-18,0.28,textr,bbox=props)
#plt.legend(loc='best')
plt.ylabel(r'$ln(t_{c^\ast})$')
plt.xlabel('ln(W)')
plt.text(-32,0.22,'(C)')
plt.legend(bbox_to_anchor=(1.0, 1.02),ncol=1,fontsize='9')
plt.show()


www2=np.arange(tc2[-1],tc2[0],0.001)
a2=0.52212501
b2=-0.12286851
y2=b2+a2*www2
plt.plot(www2,y2)
textr='\n'.join((
        r'$\delta$: '+str(delta)+' -> +'+str(-delta),
        r'L = '+str(L),
        r'$l_c(t_{c^\ast})$='+str(round(a2,4))+r'*$t_{c^\ast}$'+str(round(b2,4))))
for i in range(len(tc2)):
    plt.plot(tc2[i],rc2[i],'o')
plt.text(1.25,0.57,textr,bbox=props)
plt.show()

dy = np.zeros(len(y),np.float)
dy[0:-1] = np.diff(y)/np.diff(www)
dy[-1] = (y[-1] - y[-2])/(www[-1] - www[-2])

