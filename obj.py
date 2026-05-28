import numpy as np
import math as mt
from decimal import getcontext
# definzione della funzione obiettivo sulla base dei costi f1, f2, f3  

# dimensione = 1000
m=1000

r=100
#np.linalg.norm
def f1_i(xi,xj):
    norm= mt.dist(xi,xj)
    ret= 0.1 *(1-mt.exp(-(norm**2)))  
    return ret

def f1_tot(x):
    tot=0
    for i in range(m):
        xi = x[:,i]
        
        for j in range(i+1,m):
            tot+=f1_i(xi,x[:,j])
           # print(tot)
                
    return tot
def f2_i2(xi):
    z0=np.array([0,0])
    z1=np.array([100,100])
    z2=np.array([-100,100])
    z3=np.array([-100,-100])
    z4=np.array([100,-100])
    
    aux= (1/m) * (mt.dist(xi,z0)**2 + mt.dist(xi,z1)**2 + mt.dist(xi,z2)**2 + mt.dist(xi,z3)**2 + mt.dist(xi,z4)**2)
    return aux

LOG_DEN = np.log(1 + r**2)

def f2_i(xi):

   aux = (1/m)* (5*(mt.dist(xi,[0,0])**2) + 80000)
   return aux

def f2_tot(x):
    tot = 0
    for i in range(m):
        tot += f2_i(x[:,i])
    return tot

def s_i(i):
    
    si= (((-1)**i)* i * 0.2)
    return np.array([si,si])

def f3_i(xi,i):
    num=mt.log(1+ np.sum((xi-s_i(i))**2))
    det=mt.log(1+ 10000)
    aux=1000*((num/det) -1)**2    
    return aux

def f3_tot(x):
    tot=0
    for i in range(m):
        tot+=f3_i(x[:,i],i+1)
    return tot

def f3_tot_fast(x):
    total = 0.0
    for i in range(m):
        si = ((-1)**(i+1)) * (i+1) * 0.2
        si_vec = np.array([si, si])
        
        d2 = np.sum((x[:, i] - si_vec)**2)
        val = (np.log(1 + d2) / LOG_DEN) - 1
        total += 1000 * val**2
        
    return total


def obj_base():
    
    return 0

def obj_tot():
    
    return 0

