import subprocess
import numpy as np
import math as mt
from scipy.spatial.distance import pdist
# definzione della funzione obiettivo sulla base dei costi f1, f2, f3  

# dimensione = 1000
m=1000

r=100.0


# -------------------------------------------------------------
# f1
# -------------------------------------------------------------

def f1(x):
    """
    Calcola f1 usando NumPy e SciPy per la massima efficienza.
    
    Parametri:
    x : numpy.ndarray
        Matrice di forma (m, 2) contenente le coordinate spaziali (x_i^1, x_i^2) dei moduli.
        
    Ritorna:
    float : Il costo totale di dispersione f1.
    """
    # pdist calcola le distanze per tutte le coppie non ordinate (i < j).
    # 'sqeuclidean' calcola direttamente il quadrato della distanza euclidea: ||xi - xj||^2
    distanze_quadrate = pdist(x, metric='sqeuclidean')
    
    # Applichiamo la formula 0.1 * (1 - e^(-||xi - xj||^2)) a tutte le distanze calcolate
    dispersioni = 0.1 * (1 - np.exp(-distanze_quadrate))
    
    # f1 è la somma di tutte le dispersioni
    f1 = np.sum(dispersioni)
    return f1

# -------------------------------------------------------------
# f2
# -------------------------------------------------------------

def f2(x):
    norma = np.sum(x**2, axis=1) # norma = prodotto scalare ** 2
    aux = (1/m) * (5*norma + 80000)
    f2 = np.sum(aux)
    return f2

# -------------------------------------------------------------
# f3
# -------------------------------------------------------------

def f3(X):
    """
    Calcola la funzione di costo f3 per una matrice X di forma (m, 2).
    """
    # m è il numero di righe (moduli operativi)
    m = X.shape[0] 
    
    # 1. Creiamo gli indici i = 1, ..., m (la formula matematica parte da 1)
    I = np.arange(1, m + 1)
    
    # 2. Calcoliamo i valori della formula per le coordinate di riferimento
    # s_i = (-1)^i * i * 0.2
    valori_s = ((-1.0)**I) * I * 0.2
    
    # Costruiamo la matrice S in modo che abbia la forma (m, 2)
    # np.column_stack prende l'array 1D e lo mette in verticale come colonne
    S = np.column_stack((valori_s, valori_s))
    
    # 3. Calcoliamo la distanza al quadrato ||x_i - s_i||^2
    # Sottraiamo, eleviamo al quadrato e sommiamo lungo le colonne (axis=1)
    # In questo modo "comprimiamo" le coordinate x e y ottenendo un array di m elementi
    dist_quad = np.sum((X - S)**2, axis=1)
    
    # 4. Calcoliamo le costanti della formula
    r = 100.0
    denominatore = np.log(1 + r**2)
    
    # 5. Applichiamo la formula vettorialmente: 1000 * ((log(1 + D^2) / log(1 + r^2)) - 1)^2
    deviazioni = 1000 * ((np.log(1 + dist_quad) / denominatore) - 1.0)**2
    
    # f3 è la somma totale di tutte le deviazioni
    return np.sum(deviazioni)

# -------------------------------------------------------------
# f4
# -------------------------------------------------------------

def f4():
    # Call external executable to compute f4. Only run this if needed, as it may be time-consuming.
    result =subprocess.run([r'.\mobd.exe', 'x.txt', '-b'], capture_output=True, text=True)
    return result.stdout
    

