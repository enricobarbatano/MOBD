import numpy as np
import obj 
import gradients as grad

m=1000
x0 =np.random.uniform(-100.0,100.0,size=(m,2))  # matrice 2xm con valori casuali tra -100 e 100

np.savetxt('x.txt',x0.flatten(), fmt='%f')

# Funzioni obiettivo
if False: # set to true to compute functions
    
    f1 = obj.f1(x0)

    f2 = obj.f2(x0)

    f3 = obj.f3(x0)

    #f4 = obj.f4() #Uncomment this line if you want to compute f4, but be aware that it may take a long time to run.

    print('f1: ',f1)
    print('f2: ',f2)
    print('f3: ',f3)
    #print('f4: ',f4)

# Gradienti
print(grad.grad_f1(x0))
grad.verifica_gradiente(obj.f1, grad.grad_f1, x0)
