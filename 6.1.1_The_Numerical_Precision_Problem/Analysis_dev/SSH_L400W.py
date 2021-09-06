# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:26:34 2018

@author: LewisCYC
"""

import numpy as np
import matplotlib.pyplot as plt
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

quench015=-0.15
L=400
W=np.array([1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-6,1e-4,1e-2])
ev015=np.abs(np.array([1.84530320554e-15,2.42028537736e-14,2.97419668852e-13,3.67276818284e-12,3.14580862988e-11,2.45867619714e-10,2.50048223041e-09,2.9015470671e-07,2.8572006336e-05,0.0028000147701]))
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
print(res015)


www=np.linspace(np.log(W[0]),np.log(W[-1]),1000)
a015=1.00610552
b015=-1.17803089
y015=b015+a015*(www)
plt.plot(www,y015,'r',label='log-log fit')
textr='\n'.join((
        r'(C) $\delta$: '+str(quench015)+' -> +'+str(-quench015)+', L='+str(L),
        r'$ln(|\lambda_j|)$='+str(round(a015,4))+'*$ln(W)$'+str(round(b015,4))))
for i in range(len(W)):
    if i == len(W)-1:
        plt.plot(np.log(W[i]),np.log(ev015[i]),'o',label='W=1e-02')
    elif i == len(W)-2:
        plt.plot(np.log(W[i]),np.log(ev015[i]),'o',label='W=1e-04')
    else:
        plt.plot(np.log(W[i]),np.log(ev015[i]),'o',label='W='+str(W[i]))

plt.legend(loc='upper left',prop={'size':9})
plt.ylabel(r'$ln(|\lambda_j|)$')
plt.xlabel(r'$ln(W)$')
plt.text(-20,-30,textr,color='r',bbox=props)
plt.text(-25,-8,'(C)')
plt.show()
