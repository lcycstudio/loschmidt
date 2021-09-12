# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:26:34 2018

@author: LewisCYC
"""

import numpy as np
import matplotlib.pyplot as plt
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

quench=-0.95
L=30
W=np.array([1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-6,1e-4,1e-2])
ev=np.abs(np.array([4.99837412887e-15,4.9403664231e-14,4.81089251036e-13,5.3070563462e-12,4.84756968968e-11,4.99029496739e-10,4.94562624453e-09,4.84344443756e-07,4.63637342789e-05,0.00510926997215]))
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
a095=0.99923885
b095=-0.720827
y=b095+a095*(www)
plt.plot(www,y,'g',label='log-log fit')
textr='\n'.join((
        r'$\delta$: '+str(quench)+' -> +'+str(-quench)+', L='+str(L),
        r'$ln(|\lambda_j|)$='+str(round(a095,4))+'*$ln(W)$'+str(round(b095,4))))
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
plt.text(-20,-30,textr,color='g',bbox=props)
plt.text(-25,-7,'(A)')
plt.show()
