# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:26:34 2018

@author: LewisCYC
"""

import numpy as np
import matplotlib.pyplot as plt
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

quench05=-0.5
L=800
W=np.array([1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-6,1e-4,1e-2])
ev=np.abs(np.array([3.03383977204e-15,5.12712469453e-14,4.01670424851e-13,4.5373513352e-12,4.97523644548e-11,4.86537521723e-10,4.82964184167e-09,4.72738435458e-07,4.38315048678e-05,0.00420598207155]))
plt.plot(np.log(W),np.log(ev),'o')
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

res = least_squares(residuals, params, args=(np.log(W), np.log(ev)))
print(res)


www=np.linspace(np.log(W[0]),np.log(W[-1]),1000)
a05=1.00407652
b05=-0.73142332
y=b05+a05*(www)
plt.plot(www,y,'b',label='log-log fit')
textr='\n'.join((
        r'(B) $\delta$: '+str(quench05)+' -> +'+str(-quench05)+', L='+str(L),
        r'$ln(|\lambda_j|)$='+str(round(a05,4))+'*$ln(W)$'+str(round(b05,4))))
for i in range(len(W)):
    if i == len(W)-1:
        plt.plot(np.log(W[i]),np.log(ev[i]),'o',label='W=1e-02')
    elif i == len(W)-2:
        plt.plot(np.log(W[i]),np.log(ev[i]),'o',label='W=1e-04')
    else:
        plt.plot(np.log(W[i]),np.log(ev[i]),'o',label='W='+str(W[i]))
plt.legend(loc='upper left',prop={'size':9})
plt.ylabel(r'$ln(|\lambda_j|)$')
plt.xlabel(r'$ln(W)$')
plt.text(-20,-30,textr,color='b',bbox=props)
plt.text(-25,-7,'(B)')
plt.show()

www=np.arange(0,1000,1)
a=-0.54949162
b=0.29162733
y=b+a*(www)
plt.plot(www,y,label='log fit')
plt.legend('log-log fit')
plt.show()