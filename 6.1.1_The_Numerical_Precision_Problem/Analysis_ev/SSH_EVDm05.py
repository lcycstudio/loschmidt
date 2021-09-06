# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:26:34 2018

@author: LewisCYC
"""

import numpy as np
import matplotlib.pyplot as plt
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

quench=-0.5
size=np.array([10,20,40,60,80,100,120,140,200])
ev=np.abs(np.array([0.0054873711907937632,2.2580117134735766e-05,3.8239627870468688e-10,6.4875693121519415e-15,1.3375364094014039e-17,-9.5554098488109978e-18,1.1654979324111855e-17,1.1082080414698356e-17,-9.5554117054792474e-18]))
plt.plot(size,np.log(ev),'o')
plt.show()
size1=size[:4]
ev1=ev[:4]

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


www=np.arange(0,200,1)
a=-0.54949162
b=0.29162733
y=b+a*(www)
plt.plot(www,y,label='log fit')
textr='\n'.join((
        r'$\delta$: '+str(quench)+' -> +'+str(-quench),
        r'$ln(|\lambda_j|)$='+str(round(a,4))+'*L+'+str(round(b,4))))
for i in range(len(size)):
    plt.plot(size[i],np.log(ev[i]),'o',label='L='+str(size[i]))
plt.legend(loc='lower left')#bbox_to_anchor=(1.0, 1.02),ncol=1)#loc='best')
plt.ylabel(r'$ln(|\lambda_j|)$')
plt.xlabel('System sites L')
plt.text(115,-15,textr,bbox=props)
plt.text(37,-10,'(B)')
plt.show()

www=np.arange(0,1000,1)
a=-0.54949162
b=0.29162733
y=b+a*(www)
plt.plot(www,y,label='log fit')
plt.legend('log fit')
plt.show()