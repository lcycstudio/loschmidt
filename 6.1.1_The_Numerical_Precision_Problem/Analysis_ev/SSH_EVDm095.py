# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:26:34 2018

@author: LewisCYC
"""

import numpy as np
import matplotlib.pyplot as plt
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

quench=-0.95
size=np.array([10,20,30,40,50,60,100])
ev=np.abs(np.array([2.159861735080128e-08,2.6361724992988903e-16,1.2777951535188326e-20,-1.8687349967418192e-20,-3.0533237993358682e-20,1.277529827475995e-20,-3.0533237993358682e-20]))
plt.plot(size,np.log(ev),'o')
plt.show()
size1=size[:2]
ev1=ev[:2]

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

res = least_squares(residuals, params, args=(size1, np.log(ev1)))
print(res)


www=np.arange(0,100,1)
a=-1.82213969
b=0.57076036
y=b+a*(www)
plt.plot(www,y,label='log fit')
textr='\n'.join((
        r'$\delta$: '+str(quench)+' -> +'+str(-quench),
        r'$ln(|\lambda_j|)$='+str(round(a,4))+'*L+'+str(round(b,4))))
for i in range(len(size)):
    plt.plot(size[i],np.log(ev[i]),'o',label='L='+str(size[i]))
plt.legend(loc='lower left')
plt.ylabel(r'$ln(|\lambda_j|)$')
plt.xlabel('System sites L')
plt.text(55,-20,textr,bbox=props)
plt.text(20,-10,'(A)')
plt.show()

www=np.arange(0,1000,1)
a=-0.54949162
b=0.29162733
y=b+a*(www)
plt.plot(www,y,label='log fit')
plt.legend('log fit')
plt.show()