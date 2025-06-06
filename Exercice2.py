import numpy as np
import math

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

def matrice_stochastique(C):
    # Transforme la matrice en matrice numpy en float
    C = np.array(C, dtype=float)
    # Somme par colonne
    som_col = C.sum(axis=0)
    # Remplace 0 par 1 pour éviter division par zéro
    som_col[som_col == 0] = 1
    # Normalise chaque colonne
    # La somme de la colonne fait 1
    Q = C / som_col
    return Q

web = [
    0, 0, 0, 0, 0,
    1, 0, 0, 0, 0,
    1, 0, 0, 1, 0,
    1, 1, 0, 0, 0,
    1, 1, 0, 0, 0
]

# Appliquement du facteur d'armortissement
def matrice_transition_P(A, alpha=0.85):
    for i in range(len(A)):
        colSom = np.sum(A[:, i]) # Stockage de la somme d'une colonne
        if colSom == 0:
            A[:, i] = [1/len(A)] # Si somme = 0, alors la colonne entière devienne 1/N
        else:
            A[:, i] = alpha * A[:, i] + (1-alpha)/len(A) # Sinon, chaque cellule devienne alpha * cellule + (1-alpha)/N
    return A

def puissance_iteree_v2(C, p, alpha=0.85):
    """
    Fonction puissance_iteree qui prend en entrée 
    une matrice A, une précision p et qui retourne 
    le vecteur propre.
    
    A       matrice, doit être un tableau
    p       précision, doit être un entier   
    """

    C = np.array(C)

    # Transposé de la matrice
    C = matrice_transposee(C)

    # Calcule de la matrice Q
    Q = matrice_stochastique(C)

    # Traitement par colonne
    matrice_transition_P(Q, alpha)

    # Nombre de page
    N = Q.shape[0]

    # Vecteur initial
    r = np.ones(N) / N # Normalisation

    while True:
        ancien_r = r.copy()
        r = np.dot(Q, r)
        # On compare deux matrices selon la précision p
        if np.allclose(r, ancien_r, atol=p):
            return ancien_r, Q

precision = 1e-6

r, Q = puissance_iteree_v2(web, precision)
for page in range(len(r)):
    print(f"Page {page+1:<3}: {r[page]:.4f}")

# Vérification que r = Q*r
verification = np.dot(Q, r)
print("On vérifie que r = Qr")
if np.allclose(verification, r, atol=precision):
    print("r = Qr")
else:
    print("r != Qr")

"""
1) Si on applique l'algorithme de la partie 1 au graphe de cette partie,
celui-ci plantera parce que dans ce graphe on a un puit, ce qui produit une colonne
nulle dans la matrice. Le problème est que vu qu'on veux multiplier par un vecteur
aléatoire, cette colonne fera en sort que le rank pour ce sommet soit zero, ce qui ne
fait pas de sens car il sera forcémént pointé par quelqu'un, et il aura donc forcément
un score non nul.

2) En appliquant la nouvelle matrice de transition, on attribue donc le facteur
d'amortissement, ce qui nous permet d'attribuer aux cellules nulles une valeur
non nulle pour que le sommet correspondant aie un score même s'il s'agit d'un puit.
"""