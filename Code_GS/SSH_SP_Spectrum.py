# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import scipy.special
import numpy as np

import matplotlib.pyplot as plt



""" Dispersion relation """

k=np.linspace(-np.pi,np.pi,200)
#plt.figure(figsize=(15,2.5))
#plt.subplot(151)
v=1
w=0
e1=np.sqrt(v**2+w**2+2*v*w*np.cos(k))
e2=-np.sqrt(v**2+w**2+2*v*w*np.cos(k))
plt.plot(k,e1)
plt.plot(k,e2)
plt.yticks([-2,-1,0,1,2])
plt.xticks([-np.pi,0,np.pi],[r'$-\pi$', r'$0$', r'$\pi$'])
plt.ylabel('Energy E')
plt.xlabel('wavenumber k')
plt.text(1, 0.2, r'$w=0$')
plt.axhline(y=0,color='tab:grey',linestyle='--')
plt.axvline(x=0,color='tab:grey',linestyle='--')
plt.show()


v=1
w=0.5
e1=np.sqrt(v**2+w**2+2*v*w*np.cos(k))
e2=-np.sqrt(v**2+w**2+2*v*w*np.cos(k))
plt.plot(k,e1)
plt.plot(k,e2)
plt.yticks([-2,-1,0,1,2])
plt.xticks([-np.pi,0,np.pi],[r'$-\pi$', r'$0$', r'$\pi$'])
plt.ylabel('Energy E')
plt.xlabel('wavenumber k')
plt.text(1, 0.2, r'$v>w$')
plt.axhline(y=0,color='tab:grey',linestyle='--')
plt.axvline(x=0,color='tab:grey',linestyle='--')
plt.show()


v=1
w=1
e1=np.sqrt(v**2+w**2+2*v*w*np.cos(k))
e2=-np.sqrt(v**2+w**2+2*v*w*np.cos(k))
plt.plot(k,e1)
plt.plot(k,e2)
plt.ylim([-2,2])
plt.yticks([-2,-1,0,1,2])
plt.xticks([-np.pi,0,np.pi],[r'$-\pi$', r'$0$', r'$\pi$'])
plt.ylabel('Energy E')
plt.xlabel('wavenumber k')
plt.text(1, 0.2, r'$v=w$')
plt.axhline(y=0,color='tab:grey',linestyle='--')
plt.axvline(x=0,color='tab:grey',linestyle='--')
plt.show()


v=0.5
w=1
e1=np.sqrt(v**2+w**2+2*v*w*np.cos(k))
e2=-np.sqrt(v**2+w**2+2*v*w*np.cos(k))
plt.plot(k,e1)
plt.plot(k,e2)
plt.yticks([-2,-1,0,1,2])
plt.xticks([-np.pi,0,np.pi],[r'$-\pi$', r'$0$', r'$\pi$'])
plt.ylabel('Energy E')
plt.xlabel('wavenumber k')
plt.text(1, 0.2, r'$v<w$')
plt.axhline(y=0,color='tab:grey',linestyle='--')
plt.axvline(x=0,color='tab:grey',linestyle='--')
plt.show()


v=0
w=1
e1=np.sqrt(v**2+w**2+2*v*w*np.cos(k))
e2=-np.sqrt(v**2+w**2+2*v*w*np.cos(k))
plt.plot(k,e1)
plt.plot(k,e2)
plt.yticks([-2,-1,0,1,2])
plt.xticks([-np.pi,0,np.pi],[r'$-\pi$', r'$0$', r'$\pi$'])
plt.ylabel('Energy E')
plt.xlabel('wavenumber k')
plt.text(1, 0.2, r'$v=0$')
plt.axhline(y=0,color='tab:grey',linestyle='--')
plt.axvline(x=0,color='tab:grey',linestyle='--')
plt.show()

#plt.subplots_adjust(wspace=0.2)
#plt.show()



""" Winding number """
#k=np.linspace(0,2*np.pi,200)
#plt.figure(figsize=(15,2.5))


plt.axhline(y=0,color='tab:grey',linestyle='--')
plt.axvline(x=0,color='tab:grey',linestyle='--')
v=1
w=0
dx=v+w*np.cos(k)
dy=w*np.sin(k)
plt.plot(dx,dy,'o',color='b')
plt.axis('equal')
plt.yticks([-2,0,2])
plt.xticks([-1,0,2])
plt.ylabel('$d_y$(k)')
plt.xlabel('$d_x$(k)')
plt.show()


plt.axhline(y=0,color='tab:grey',linestyle='--')
plt.axvline(x=0,color='tab:grey',linestyle='--')
v=1
w=0.5
dx=v+w*np.cos(k)
dy=w*np.sin(k)
plt.plot(dx,dy,'b')
plt.axis('equal')
plt.yticks([-1,0,1])
plt.xticks([-1,0,2])
plt.ylabel('$d_y$(k)')
plt.xlabel('$d_x$(k)')
plt.arrow(1.4, 0.17, -0.01, 0.012, head_width=0.1, head_length=0.2, fc='b', ec='b')
plt.show()


plt.axhline(y=0,color='tab:grey',linestyle='--')
plt.axvline(x=0,color='tab:grey',linestyle='--')
v=1
w=1
dx=v+w*np.cos(k)
dy=w*np.sin(k)
plt.plot(dx,dy,'b')
plt.axis('equal')
plt.yticks([-1,0,1])
plt.xticks([-1,0,2])
plt.ylabel('$d_y$(k)')
plt.xlabel('$d_x$(k)')
plt.arrow(1.87, 0.5, -0.01, 0.013, head_width=0.2, head_length=0.3, fc='b', ec='b')
plt.show()


plt.axhline(y=0,color='tab:grey',linestyle='--')
plt.axvline(x=0,color='tab:grey',linestyle='--')
v=0.5
w=1
dx=v+w*np.cos(k)
dy=w*np.sin(k)
plt.plot(dx,dy,'b')
plt.axis('equal')
plt.yticks([-1,0,1])
plt.xticks([-1,0,2])
plt.ylabel('$d_y$(k)')
plt.xlabel('$d_x$(k)')
plt.arrow(1.45, 0.3, -0.01, 0.02, head_width=0.2, head_length=0.3, fc='b', ec='b')
plt.show()


plt.axhline(y=0,color='tab:grey',linestyle='--')
plt.axvline(x=0,color='tab:grey',linestyle='--')
v=0
w=1
dx=v+w*np.cos(k)
dy=w*np.sin(k)
plt.plot(dx,dy,'b')
plt.axis('equal')
plt.yticks([-1,0,1])
plt.xticks([-1,0,1])
plt.ylabel('$d_y$(k)')
plt.xlabel('$d_x$(k)')
plt.arrow(0.95, 0.3, -0.01, 0.021, head_width=0.2, head_length=0.3, fc='b', ec='b')
plt.show()

#plt.subplots_adjust(wspace=0.2)


""" Adiabatically connected """
plt.axhline(y=0,color='tab:grey',linestyle='--')
plt.axvline(x=0,color='tab:grey',linestyle='--')
v=1
w=1
vg=np.linspace(0,v,10)
b=0
for i in vg:
    dx=i+w*np.cos(k)
    dy=w*np.sin(k)
    if b==0:
        plt.plot(dx,dy,'b')
    elif b==9:
        plt.plot(dx,dy,'b')
    else:
        plt.plot(dx,dy,'b--')
    b=b+1
plt.ylim([-1.5,1.5])
plt.xlim([-1.2,2.5])
plt.text(2.3, -0.3, r'$d_x$')
plt.text(0.1, 1.3, r'$d_y$')
plt.arrow(0.0,1.0,1, 0, head_width=0.1, head_length=0.2, fc='r', ec='r')
plt.arrow(0.0,-1.0,1, 0, head_width=0.1, head_length=0.2, fc='r', ec='r')
plt.arrow(1.0,0,-0.01, 0.09, head_width=0.1, head_length=0.2, fc='b', ec='b')
plt.arrow(2.0,0,-0.01, 0.09, head_width=0.1, head_length=0.2, fc='b', ec='b')
plt.show()


plt.axhline(y=0,color='tab:grey',linestyle='--')
plt.axvline(x=0,color='tab:grey',linestyle='--')
v=0.8
w=1
wg=np.linspace(0,w,10)
c=0
for i in vg:
    dx=v+i*np.cos(k)
    dy=i*np.sin(k)
    if c==0:
        plt.plot(dx,dy,'b')
    elif c>8:
        plt.plot(dx,dy,'b')
    else:
        plt.plot(dx,dy,'b--')
    c=c+1#plt.arrow(0.95, 0.3, -0.01, 0.021, head_width=0.2, head_length=0.3, fc='b', ec='b')
plt.ylim([-1.2,1.2])
plt.xlim([-1.2,2.0])
plt.text(1.9, -0.2, r'$d_x$')
plt.text(0.1, 1.0, r'$d_y$')
plt.arrow(0.8,0.01,0, 0.8, head_width=0.1, head_length=0.2, fc='r', ec='r')
plt.arrow(0.8,-0.01,0, -0.8, head_width=0.1, head_length=0.2, fc='r', ec='r')
plt.arrow(1.8,0,-0.01, 0.09, head_width=0.1, head_length=0.2, fc='b', ec='b')
plt.show()


#plt.axhline(y=0,color='tab:grey',linestyle='--')
k=np.linspace(-np.pi,np.pi,200)
w=1
theta=np.linspace(0,np.pi,20)
vg=np.linspace(0,1,10)
fig = plt.figure()
ax = plt.axes(projection='3d')
for i in theta:
    v=1.5-np.cos(i)
    u=np.sin(i)
    dx=v+w*np.cos(k)
    dy=w*np.sin(k)
    ax.plot(dx, dy, u)
    #u=np.sin(theta)
ax.set_xticks([0.5,1,1.5])
ax.set_yticks([])    
ax.set_zticks([])
ax.set_xlabel('dy')
ax.set_ylabel('dx')
ax.set_zlabel('dz')
