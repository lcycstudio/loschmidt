# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 09:44:14 2018

@author: LewisCYC
"""

import numpy as np
#from bisect import bisect_left as findIndex

def manyPsi(particle,site):
    a=np.arange(2**site)
    bitcount=np.array([bin(x).count("1") for x in a])
    b=a.compress(bitcount==particle).tolist()[::-1]
    aList=[]
    for item in b:
        a=bin(int(item))[2:].zfill(site)
        aList.append(a)
    return aList

Psi=manyPsi(2,4)

def cpc(l,j,Psi):
    Psi2=Psi.copy()
    for item in Psi2:
        k = Psi2.index(item)
        if item[j-1]=='0':
            Psi2[k]=list(item)
            Psi2[k][j-1]='1'
            Psi2[k]="".join(Psi2[k])
        if item[j-1]=='1':
            Psi2[k]=list(item)
            Psi2[k][j-1]='0'
            Psi2[k]="".join(Psi2[k])
    for item in Psi2:
        k = Psi2.index(item)
        if item[l-1]=='0':
            Psi2[k]=list(item)
            Psi2[k][l-1]='1'
            Psi2[k]="".join(Psi2[k])
        if item[l-1]=='1':
            Psi2[k]=list(item)
            Psi2[k][l-1]='0'
            Psi2[k]="".join(Psi2[k])
    return Psi2
            
Psi2=cpc(1,2,Psi)
Psi=manyPsi(2,4)

def buildSSH(delta,site):
    particle=site/2
    Psi=manyPsi(particle,site)
    L=len(Psi)
    H=np.zeros((L,L))
    for a in range(1,site):
        b=a+1
        Psi2=cpc(a,b,Psi)
        for j in Psi2:
            jj=Psi2.index(j)
            for i in Psi:
                ii=Psi.index(i)
                if i == j:
                    H[ii][jj]=1-delta*(-1)**a
    return H

H=buildSSH(0,4)