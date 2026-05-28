import numpy as np
import obj 
import matplotlib.pyplot as plt

m=1000
x0 =np.random.uniform(-100.0,100.0,size=(2,m))  # matrice 2xm con valori casuali tra -100 e 100
print(x0)
np.savetxt('x.txt',x0.flatten(), fmt='%f')
f1= obj.f1_tot(x0)

f2= obj.f2_tot(x0)
f3= obj.f3_tot_fast(x0)
print(f1)
print(f2)
print(f3)

#12921.293207804929