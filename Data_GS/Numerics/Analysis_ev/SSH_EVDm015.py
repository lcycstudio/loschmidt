# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:26:34 2018

@author: LewisCYC
"""

import numpy as np
import matplotlib.pyplot as plt
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

quench=-0.15
size=np.array([20,60,100,150,200,250,300,400])
ev=np.abs(np.array([0.0256363206325,6.01292859063e-05,1.42399241414e-07,7.43934133166e-11,3.81046038821e-14,-1.8041290995e-16,-5.59398295738e-17,-7.60689114308e-16]))
plt.plot(size,np.log(ev),'o')
plt.show()
size1=size[:5]
ev1=ev[:5]

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


www=np.arange(0,400,1)
a=-0.15127272
b=-0.63858824
y=b+a*(www)
plt.plot(www,y,label='log fit')
textr='\n'.join((
        r'$\delta$: '+str(quench)+' -> +'+str(-quench),
        r'$ln(|\lambda_j|)$='+str(round(a,4))+'*L'+str(round(b,4))))
for i in range(len(size)):
    plt.plot(size[i],np.log(ev[i]),'o',label='L='+str(size[i]))
plt.legend(loc='lower left')#bbox_to_anchor=(1.0, 1.02),ncol=1)#loc='best')
plt.ylabel(r'$ln(|\lambda_j|)$')
plt.xlabel('System sites L')
plt.text(235,-10,textr,bbox=props)
plt.text(75,-7,'(C)')
plt.show()

www=np.arange(0,1000,1)
a=-0.54949162
b=0.29162733
y=b+a*(www)
plt.plot(www,y,label='log fit')
plt.legend('log fit')
plt.show()