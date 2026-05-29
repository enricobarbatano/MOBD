import numpy as np
import csv
from pathlib import Path

from createXfor_dataset import phase1_gauss_seidel_armijo, M

# ============================================================
# CONFIG
# ============================================================

WORKDIR = Path(r"C:\Users\User\OneDrive\Desktop\Progetti_Uni\PROGETTO_MOBD\progetto")
OUTDIR = WORKDIR / "phase1_out"
OUTDIR.mkdir(parents=True, exist_ok=True)

BEST_X_FILE = WORKDIR / "best_x_strategy4.txt"   # se esiste lo usiamo come punto iniziale
SNAPSHOT_NPZ = OUTDIR / "phase1_snapshots.npz"
HISTORY_CSV  = OUTDIR / "phase1_history.csv"

# ============================================================
# PARAMETRI FASE 1
# ============================================================

EPOCHS = 5                 # quante epoche (passate su tutti i moduli)
SAVE_EVERY_UPDATES = 200   # ogni quanti update salvo uno snapshot
SEED = 42

# Armijo (come tipicamente usato negli esempi del corso) 【1-48fd6f】
ALPHA0 = 1.0
GAMMA  = 1e-2
DELTA  = 0.5

# ============================================================
# UTILS
# ============================================================

def load_initial_X() -> np.ndarray:
    """
    Legge un file x.txt / best_x_strategy4.txt con 2000 valori (flatten) e lo riconverte in (2, M).
    Se non esiste, genera uno start casuale in [-100,100].
    """
    if BEST_X_FILE.exists():
        v = np.loadtxt(BEST_X_FILE, dtype=np.float64)
        if v.size != 2 * M:
            raise ValueError(f"{BEST_X_FILE} contiene {v.size} valori, attesi {2*M}")
        # file è tipicamente 2000 righe: x0,y0,x1,y1,... oppure una colonna lunga
        # Noi lo interpretiamo come (2,M) con ordine: [x0,y0,x1,y1,...]
        X = v.reshape(M, 2).T  # (2,M)
        return X
    else:
        rng = np.random.default_rng(SEED)
        return rng.uniform(-100.0, 100.0, size=(2, M)).astype(np.float64)

def save_history_csv(history, path: Path) -> None:
    """
    Scrive una lista di dict in CSV anche se i dict hanno chiavi diverse.
    - fieldnames = unione di tutte le chiavi
    - extrasaction='ignore' evita errori se arrivano chiavi nuove
    - converte np scalari in tipi Python (float/int) per scrittura pulita
    """
    if not history:
        return

    # unione delle chiavi
    fieldnames = sorted({k for row in history for k in row.keys()})

    def normalize_value(v):
        # converte np.float64 / np.int64 ecc. in float/int Python
        if hasattr(v, "item"):
            try:
                return v.item()
            except Exception:
                return v
        return v

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in history:
            clean_row = {k: normalize_value(v) for k, v in row.items()}
            w.writerow(clean_row)


# ============================================================
# MAIN
# ============================================================

def main():
    X0 = load_initial_X()
    print("[INFO] X0 loaded:", X0.shape)

    X, history, snapshots = phase1_gauss_seidel_armijo(
        X0,
        epochs=EPOCHS,
        alpha0=ALPHA0,
        gamma=GAMMA,
        delta=DELTA,
        save_every_updates=SAVE_EVERY_UPDATES,
        seed=SEED
    )

    print("[INFO] Phase1 complete.")
    print("[INFO] Final X shape:", X.shape)
    print("[INFO] Num snapshots:", len(snapshots))
    if history:
        print("[INFO] Last history row:", history[-1])

    # salva snapshots in npz: (N,2,M)
    X_stack = np.stack(snapshots, axis=0)
    np.savez(SNAPSHOT_NPZ, X=X_stack)
    print("[INFO] Saved snapshots to:", SNAPSHOT_NPZ)

    # salva history in CSV
    save_history_csv(history, HISTORY_CSV)
    print("[INFO] Saved history to:", HISTORY_CSV)

    # salva anche il punto finale (flatten) se vuoi riusarlo
    final_txt = OUTDIR / "phase1_final_x.txt"
    np.savetxt(final_txt, X.T.reshape(-1), fmt="%.15f")
    print("[INFO] Saved final x to:", final_txt)


if __name__ == "__main__":
    main()