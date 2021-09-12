# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:26:34 2018

@author: LewisCYC
"""

import numpy as np
import matplotlib.pyplot as plt
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

quench015=-0.15
quench05=-0.5
quench095=-0.95
W=np.array([1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-6,1e-4,1e-2])
ev015=np.abs(np.array([1.84530320554e-15,2.42028537736e-14,2.97419668852e-13,3.67276818284e-12,3.14580862988e-11,2.45867619714e-10,2.50048223041e-09,2.9015470671e-07,2.8572006336e-05,0.0028000147701]))
ev05=np.abs(np.array([3.03383977204e-15,5.12712469453e-14,4.01670424851e-13,4.5373513352e-12,4.97523644548e-11,4.86537521723e-10,4.82964184167e-09,4.72738435458e-07,4.38315048678e-05,0.00420598207155]))
ev095=np.abs(np.array([4.99837412887e-15,4.9403664231e-14,4.81089251036e-13,5.3070563462e-12,4.84756968968e-11,4.99029496739e-10,4.94562624453e-09,4.84344443756e-07,4.63637342789e-05,0.00510926997215]))
plt.plot(np.log(W),np.log(ev015),'o')
plt.show()


from scipy.optimize import least_squares


params = np.array([1,1])

def funcinv(x, a, b):
    return b+a*x

def residuals(params, x, data):
    # evaluates function given vector of params [a, b]
    # and return residuals: (observed_data - model_data)
    a, b = params
    func_eval = funcinv(x, a, b)
    return (data - func_eval)

res015 = least_squares(residuals, params, args=(np.log(W), np.log(ev015)))
res05 = least_squares(residuals, params, args=(np.log(W), np.log(ev05)))
res095 = least_squares(residuals, params, args=(np.log(W), np.log(ev095)))
print(res015)
print(res05)
print(res095)

www=np.linspace(np.log(W[0]),np.log(W[-1]),1000)
a015=1.00410552
b015=-1.17803089
a05=1.00407652
b05=-0.73142332
a095=0.99923885
b095=-0.720827
y015=b015+a015*(www)
y05=b05+a05*(www)
y095=b095+a095*(www)
plt.plot(www,y015,'r',label=r'$|\delta|$='+str(-quench015))
plt.plot(www,y05,'b',label=r'$|\delta|$='+str(-quench05))
plt.plot(www,y095,'g',label=r'$|\delta|$='+str(-quench095))
textr015='\n'.join((
        #r'$\delta$: '+str(quench)+' -> +'+str(-quench),
        r'$ln(|\lambda^\ast_j|)$='+str(round(a015,4))+'*$ln(W)$'+str(round(b015,4))+', L=30',))
textr05='\n'.join((
        #r'$\delta$: '+str(quench)+' -> +'+str(-quench),
        r'$ln(|\lambda^\ast_j|)$='+str(round(a05,4))+'*$ln(W)$'+str(round(b05,4))+', L=800',))
textr095='\n'.join((
        #r'$\delta$: '+str(quench)+' -> +'+str(-quench),
        r'$ln(|\lambda^\ast_j|)$='+str(round(a095,4))+'*$ln(W)$'+str(round(b095,4))+', L=400',))
for i in range(len(W)):
    if i == len(W)-1:
        plt.plot(np.log(W[i]),np.log(ev015[i]),'o',label='W=1e-02')
    elif i == len(W)-2:
        plt.plot(np.log(W[i]),np.log(ev015[i]),'o',label='W=1e-04')
    else:
        plt.plot(np.log(W[i]),np.log(ev015[i]),'o',label='W='+str(W[i]))
        
for i in range(len(W)):
    if i == len(W)-1:
        plt.plot(np.log(W[i]),np.log(ev05[i]),'o')
    elif i == len(W)-2:
        plt.plot(np.log(W[i]),np.log(ev05[i]),'o')
    else:
        plt.plot(np.log(W[i]),np.log(ev05[i]),'o')
for i in range(len(W)):
    if i == len(W)-1:
        plt.plot(np.log(W[i]),np.log(ev095[i]),'o')
    elif i == len(W)-2:
        plt.plot(np.log(W[i]),np.log(ev095[i]),'o')
    else:
        plt.plot(np.log(W[i]),np.log(ev095[i]),'o')        
plt.legend(bbox_to_anchor=(1.0, 1.02),ncol=1)#loc='best')
plt.ylabel(r'$ln(|\lambda_j|)$')
plt.xlabel(r'$ln(W)$')
plt.ylim(-40,-3)
plt.text(-22,-28,textr015,color='r',bbox=props)
plt.text(-22,-32,textr05,color='b',bbox=props)
plt.text(-22,-36,textr095,color='g',bbox=props)
plt.show()
