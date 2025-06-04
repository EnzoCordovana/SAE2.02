import numpy as np
import math

# np.set_printoptions(precision=1, linewidth=np.inf) type: ignore

def matrice_transposee(A):
    A = np.array(A)
    if A.ndim == 1:
        # Calcule la dimension de la matrice
        d = int(np.sqrt(len(A)))
        # Redimensionner la matrice en une matrice carrée
        # [a,b,c,d] => [[a,b],[c,d]]
        A = A.reshape((d, d))
    # Retourne directement la transposée
    return A.T

def matrice_stochastique(A):
    # Transforme la matrice en matrice numpy en float
    A = np.array(A, dtype=float)
    # Somme par colonne
    som_col = A.sum(axis=0)
    # Remplace 0 par 1 pour éviter division par zéro
    som_col[som_col == 0] = 1
    # Normalise chaque colonne
    # La somme de la colonne fait 1
    Q = A / som_col
    return Q

def norme(X):
    """
    Fonction norme qui prend en entrée 
    un vecteur X et qui calcule sa norme.

    Arguments:
    X   vecteur, doit être un tableau
    """
    sum = 0
    # On ajoute le carré à la variable
    for x in np.array(X):
        sum += x**2
    # On retourne la racine à la variable
    return np.sqrt(sum)


def puissance_iteree(A, p):
    """
    Fonction puissance_iteree qui prend en entrée 
    une matrice A, une précision p et qui retourne 
    la valeur propre ainsi que le vecteur propre.
    
    A       matrice, doit être un tableau
    p       précision, doit être un entier   
    """
    A = np.array(A)

    # Dimension de la matrice carré
    d = int(math.sqrt(len(A)))

    # Vecteur X_O \in R^N
    vec_0 = np.array([])
    for _ in range(d):
        vec_0 = np.append(vec_0, np.random.randint(-14, 14))

    # Redimensionner la matrice en une matrice carrée
    # [a,b,c,d] => [[a,b],[c,d]]
    A = A.reshape((d, d))

    lambda0 = 0
    lambda1 = 0

    while True:
        # Matrice * vecteur
        AX = np.dot(A, vec_0)

        # Valeur propre
        lambda0 = lambda1
        lambda1 = norme(AX)

        # Vecteur propre
        vec_0 = AX / lambda1

        # Condition de sortie
        if abs(lambda1  -lambda0) < p:
            return lambda1, vec_0

# 1. L'algorithme de puissance itérée permet de calculer le score de chacune des pages car
# on attribue aléatoirement un score à chaque page puis on va itérér jusqu'à ce que la valeur propre change pratiquement pas.

# 2.
# Graphe de la partie 2 pointant i vers j
# Web_{i,j}

web = [
    0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
]

def puissance_iteree_v2(A, p, alpha):
    """
    Fonction puissance_iteree qui prend en entrée 
    une matrice A, une précision p et qui retourne 
    le vecteur propre.
    
    A       matrice, doit être un tableau
    p       précision, doit être un entier   
    """

    A = np.array(A)

    # Transpose et stochastique la matrice
    A = matrice_stochastique(matrice_transposee(A))
    
    print(A)

    for i in range(len(A)):
        colSom = np.sum(A[:, i])
        if colSom == 0:
            A[:, i] = [1/len(A)]
        else:
            A[:, i] = alpha * A[:, i] + (1-alpha)/len(A)

    print(A)

    d = A.shape[0]
    r = np.ones(d) / d # Normalisation
    iterations = 0

    while True:
        ancien_r = r
        r = np.dot(A, r)
        iterations += 1
        # On compare deux matrices selon la précision p
        if np.allclose(r, ancien_r, atol=p):
            print(f"Convergence atteinte en {iterations} itérations avec p = {p}")
            return ancien_r

r = puissance_iteree_v2(web, 1e-6, 0.85)

# On crée une liste de tuples (index_page, score)
page_rank = list(enumerate(r, start=1))

print("PageRank :")
precisions = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

for page, rank in page_rank:
    print(f"Page {page}\t: {rank:.4f}")

for p in precisions:
    print(f"\nCalcul avec précision {p}:")
    rH = puissance_iteree_v2(web, p, 0.85)