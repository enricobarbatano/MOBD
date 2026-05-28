import numpy as np

def generate_random_input(filename="x.txt", N=1000, low=-200, high=200):
    """
    Genera 1000 punti 2D casuali
    range: uniforme tra low e high
    """
    # 2000 valori
    X = np.random.uniform(low, high, size=2*N)

    with open(filename, "w") as f:
        for val in X:
            f.write(f"{val}\n")

    return X

# esempio
X = generate_random_input("x.txt")
print("File x.txt generato!")
