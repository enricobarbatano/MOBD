import numpy as np

M = 1000
LOWER, UPPER = -300.0, 300.0
r = 100.0
LOG_DEN = np.log(1.0 + r*r)

# ------------------------------------------------------------
# Precompute s_i per f3: shape (2, M)
# s_i = ((-1)^i * i*0.2, (-1)^i * i*0.2) con i=1..M
# ------------------------------------------------------------
def build_s_matrix(M: int = 1000) -> np.ndarray:
    idx = np.arange(1, M + 1, dtype=np.float64)
    sign = np.where(idx % 2 == 1, -1.0, 1.0)  # (-1)^i con i=1..M
    s = sign * idx * 0.2
    S = np.vstack([s, s])  # (2,M)
    return S

S_MAT = build_s_matrix(M)

# ------------------------------------------------------------
# f2: formula semplificata (come nel tuo codice):
# sum_{q=0..4} ||xi - zq||^2 = 5||xi||^2 + 80000, perché z1..z4 simmetrici
# f2_tot = (1/m) * sum_i (5||xi||^2 + 80000)
# ------------------------------------------------------------
def f2_i(xi: np.ndarray) -> float:
    return (1.0 / M) * (5.0 * float(xi @ xi) + 80000.0)

def f2_tot(X: np.ndarray) -> float:
    # X shape (2,M)
    norms2 = np.sum(X * X, axis=0)         # (M,)
    return (1.0 / M) * (5.0 * float(np.sum(norms2)) + 80000.0 * M)

def grad_f2_block(xi: np.ndarray) -> np.ndarray:
    # derivata di (1/M)*5||xi||^2 = (10/M) xi
    return (10.0 / M) * xi

# ------------------------------------------------------------
# f3: vectorized per-modulo
# f3_i = 1000 * ( log(1+||xi-si||^2)/LOG_DEN - 1 )^2
# grad f3_i = 4000 * t * (xi-si) / (LOG_DEN*(1+d2)), con t = log(1+d2)/LOG_DEN - 1
# ------------------------------------------------------------
def f3_i(xi: np.ndarray, si: np.ndarray) -> float:
    d = xi - si
    d2 = float(d @ d)
    t = (np.log(1.0 + d2) / LOG_DEN) - 1.0
    return 1000.0 * t * t

def f3_tot(X: np.ndarray) -> float:
    D = X - S_MAT
    d2 = np.sum(D * D, axis=0)
    t = (np.log(1.0 + d2) / LOG_DEN) - 1.0
    return float(np.sum(1000.0 * t * t))

def grad_f3_block(i: int, xi: np.ndarray) -> np.ndarray:
    si = S_MAT[:, i]
    d = xi - si
    d2 = float(d @ d)
    t = (np.log(1.0 + d2) / LOG_DEN) - 1.0
    return (4000.0 * t / (LOG_DEN * (1.0 + d2))) * d

# ------------------------------------------------------------
# f1: coppie (i<j): 0.1*(1-exp(-||xi-xj||^2))
# Per Fase 1 serve:
# - f1_tot (una volta all'inizio)
# - delta_f1 quando muovi solo xi
# - grad_f1 su blocco i: 0.2 * sum_{j!=i} exp(-d2_ij) * (xi - xj)
# ------------------------------------------------------------
def f1_tot(X: np.ndarray) -> float:
    # calcolo O(M^2) una tantum (1000x1000 ok)
    G = X.T @ X                     # (M,M)
    norms = np.diag(G)              # (M,)
    d2 = norms[:, None] + norms[None, :] - 2.0 * G
    # prendi solo i<j
    iu = np.triu_indices(M, k=1)
    vals = 0.1 * (1.0 - np.exp(-d2[iu]))
    return float(np.sum(vals))

def delta_f1_for_move(X: np.ndarray, i: int, xi_old: np.ndarray, xi_new: np.ndarray) -> float:
    # delta su tutte le coppie (i,j), j!=i
    # O(M) vettoriale
    Xothers = X.copy()
    Xothers[:, i] = xi_old  # assicurati X rappresenti lo stato "old" per j
    dif_old = xi_old[:, None] - Xothers
    d2_old = np.sum(dif_old * dif_old, axis=0)
    dif_new = xi_new[:, None] - Xothers
    d2_new = np.sum(dif_new * dif_new, axis=0)

    # escludi j=i
    mask = np.ones(M, dtype=bool)
    mask[i] = False

    term_old = 0.1 * (1.0 - np.exp(-d2_old[mask]))
    term_new = 0.1 * (1.0 - np.exp(-d2_new[mask]))
    return float(np.sum(term_new - term_old))

def grad_f1_block(X: np.ndarray, i: int) -> np.ndarray:
    xi = X[:, i]
    dif = xi[:, None] - X           # (2,M)
    d2 = np.sum(dif * dif, axis=0)  # (M,)
    w = np.exp(-d2)
    w[i] = 0.0
    # 0.2 * sum_j w_j * (xi - xj)
    return 0.2 * np.sum(dif * w[None, :], axis=1)

# ------------------------------------------------------------
# g = f1 + f2 + f3
# Manteniamo anche (f1,f2,f3) separati per aggiornamenti delta.
# ------------------------------------------------------------
def g_components_init(X: np.ndarray):
    f1 = f1_tot(X)
    f2 = f2_tot(X)
    f3 = f3_tot(X)
    return f1, f2, f3

def g_value_from_components(f1: float, f2: float, f3: float) -> float:
    return f1 + f2 + f3

# ------------------------------------------------------------
# Armijo on block: valuta g_trial con aggiornamento incrementale
# ------------------------------------------------------------
def block_armijo_step(X: np.ndarray, i: int,
                      f1: float, f2: float, f3: float,
                      alpha0: float = 1.0, gamma: float = 1e-2, delta: float = 0.5,
                      alpha_min: float = 1e-12,
                      max_ls: int = 30):
    """
    Un passo di Gauss-Seidel su blocco i con Armijo:
    - direzione d_i = -grad_i g
    - Armijo su g usando aggiornamenti delta, senza ricalcolare tutto
    Ritorna: (X_new, f1_new, f2_new, f3_new, accepted(bool))
    """

    xi_old = X[:, i].copy()

    # grad blocco (2D)
    g1 = grad_f1_block(X, i)
    g2 = grad_f2_block(xi_old)
    g3 = grad_f3_block(i, xi_old)
    grad_block = g1 + g2 + g3

    # se grad piccolo, skip
    if float(np.max(np.abs(grad_block))) < 1e-10:
        return X, f1, f2, f3, False

    d = -grad_block
    gd = float(grad_block @ d)  # = -||grad||^2 <= 0

    g_curr = f1 + f2 + f3

    alpha = alpha0
    accepted = False

    for _ in range(max_ls):
        xi_new = np.clip(xi_old + alpha * d, LOWER, UPPER)

        # delta f2 e f3 sono O(1)
        df2 = f2_i(xi_new) - f2_i(xi_old)
        df3 = f3_i(xi_new, S_MAT[:, i]) - f3_i(xi_old, S_MAT[:, i])

        # delta f1 è O(M)
        df1 = delta_f1_for_move(X, i, xi_old, xi_new)

        g_trial = g_curr + df1 + df2 + df3

        # Armijo condition: g(x+αd) <= g(x) + γ α grad^T d
        if g_trial <= g_curr + gamma * alpha * gd:
            accepted = True
            # apply update
            X[:, i] = xi_new
            f1 += df1
            f2 += df2
            f3 += df3
            break

        alpha *= delta
        if alpha < alpha_min:
            break

    return X, f1, f2, f3, accepted

# ------------------------------------------------------------
# Main routine: Gauss-Seidel epochs + shuffle + save iterates
# ------------------------------------------------------------
def phase1_gauss_seidel_armijo(X0: np.ndarray,
                               epochs: int = 5,
                               alpha0: float = 1.0, gamma: float = 1e-2, delta: float = 0.5,
                               save_every_updates: int = 200,
                               seed: int = 42):
    """
    Ritorna:
      X, history (list of dict), snapshots (list of X copies for dataset)
    """
    rng = np.random.default_rng(seed)

    X = X0.astype(np.float64, copy=True)
    f1, f2, f3 = g_components_init(X)

    history = []
    snapshots = []
    updates = 0

    for ep in range(epochs):
        perm = rng.permutation(M)  # shuffle Gauss-Seidel (consigliato in pratica) 【1-279cd4】
        acc = 0

        t0 = time.time() if False else None  # opzionale
        for i in perm:
            X, f1, f2, f3, ok = block_armijo_step(
                X, i, f1, f2, f3,
                alpha0=alpha0, gamma=gamma, delta=delta
            )
            updates += 1
            acc += int(ok)

            if updates % save_every_updates == 0:
                snapshots.append(X.copy())
                history.append({
                    "epoch": ep,
                    "updates": updates,
                    "g": f1 + f2 + f3,
                    "f1": f1, "f2": f2, "f3": f3,
                    "accepted_in_epoch_so_far": acc
                })

        # snapshot a fine epoca
        snapshots.append(X.copy())
        history.append({
            "epoch": ep,
            "updates": updates,
            "g": f1 + f2 + f3,
            "f1": f1, "f2": f2, "f3": f3,
            "accepted_in_epoch": acc
        })

    return X, history, snapshots