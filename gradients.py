import numpy as np
from scipy.spatial.distance import pdist

# -------------------------------------------------------------
# f1
# -------------------------------------------------------------

def grad_f1_i(X, i):
    """
    Calcola il gradiente di f1 rispetto a x.
    
    Parametri:
    x : numpy.ndarray
        Matrice di forma (m, 2) contenente le coordinate spaziali (x_i^1, x_i^2) dei moduli.
        
    Ritorna:
    numpy.ndarray : Matrice di forma (m, 2) contenente il gradiente di f1 rispetto a x.
    """

    xi = X[i]

    # Selezioniamo le righe per j da 2 a 1000 (in Python, indici da 1 a 999 compresi)
    X_j = np.delete(X, i, axis=0)

    # 2. Calcoliamo la differenza (xi - xj) per tutti i j simultaneamente
    # Grazie al broadcasting, xi (1x2) viene sottratto a ogni riga di X_j (999x2)
    diff = xi - X_j 

    # 3. Calcoliamo la norma al quadrato ||xi - xj||^2 per ogni riga
    # Eleviamo al quadrato le differenze e sommiamo per ogni riga (axis=1)
    norm_sq = np.sum(diff**2, axis=1) 

    # 4. Calcoliamo il termine esponenziale: exp(-norma^2)
    exp_term = np.exp(-norm_sq) 

    # 5. Moltiplichiamo per i coefficienti: 0.1 * 2 = 0.2
    # Aggiungiamo np.newaxis per far diventare l'array 1D (999,) una colonna (999, 1), 
    # permettendo di moltiplicarlo correttamente con la matrice diff (999x2)
    pesi = 0.2 * exp_term[:, np.newaxis]

    # 6. Moltiplichiamo il termine scalare calcolato per i vettori (xi - xj)
    # e infine facciamo la sommatoria lungo le colonne (axis=0)
    risultato = np.sum(pesi * diff, axis=0)

    return risultato

def grad_f1(X):
    """
    Calcola il gradiente di f1 rispetto a x per tutti i moduli.
    
    Parametri:
    x : numpy.ndarray
        Matrice di forma (m, 2) contenente le coordinate spaziali (x_i^1, x_i^2) dei moduli.
        
    Ritorna:
    numpy.ndarray : Matrice di forma (m, 2) contenente il gradiente di f1 rispetto a x per tutti i moduli.
    """
    m = X.shape[0]
    gradiente = np.zeros_like(X)

    for i in range(m):
        gradiente[i] = grad_f1_i(X, i)

    return gradiente

# -------------------------------------------------------------
# f2
# -------------------------------------------------------------


# -------------------------------------------------------------
# Verifica
# -------------------------------------------------------------

def verifica_gradiente(f, grad_f, X, epsilon=1e-5, tolleranza=1e-4):
    """
    Verifica il gradiente usando le differenze finite centrali.
    """
    gradiente_analitico = grad_f(X)
    gradiente_numerico = np.zeros_like(X)
    
    # Iteriamo su ogni elemento della matrice X
    it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        valore_originale = X[idx]
        
        # Fai un piccolo passo in avanti (X + epsilon)
        X[idx] = valore_originale + epsilon
        f_avanti = f(X)
        
        # Fai un piccolo passo indietro (X - epsilon)
        X[idx] = valore_originale - epsilon
        f_indietro = f(X)
        
        # Ripristina il valore originale
        X[idx] = valore_originale
        
        # Calcola la derivata usando le differenze centrali
        # formula: (f(x + e) - f(x - e)) / (2*e)
        derivata_numerica = (f_avanti - f_indietro) / (2 * epsilon)
        gradiente_numerico[idx] = derivata_numerica
        
        it.iternext()
        
    # Calcoliamo la differenza (errore) tra i due gradienti
    errore_massimo = np.max(np.abs(gradiente_analitico - gradiente_numerico))
    
    print("--- Risultati del Test del Gradiente ---")
    print(f"Errore massimo assoluto: {errore_massimo:.2e}")
    
    if errore_massimo < tolleranza:
        print("✅ SUCCESSO: Il gradiente analitico corrisponde a quello numerico!")
    else:
        print("❌ FALLIMENTO: I gradienti differiscono. C'è un errore nell'implementazione.")
        print("\nPrimi 5 elementi del gradiente Analitico:\n", gradiente_analitico[:5])
        print("\nPrimi 5 elementi del gradiente Numerico:\n", gradiente_numerico[:5])