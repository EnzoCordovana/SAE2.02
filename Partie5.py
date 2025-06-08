import numpy as np
import math
import time
import matplotlib as plt

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
    X   vecteur, doit être un tableau.
    """
    sum = 0
    # On ajoute le carré à la variable
    for x in np.array(X):
        sum += x**2
    # On retourne la racine à la variable
    return np.sqrt(sum)

def matrice_transition_P(A, alpha):
    A = np.array(A)
    # dimensiond e la matrice
    N = A.shape[0]
    for i in range(N):
        colSom = np.sum(A[:, i]) # Stockage de la somme d'une colonne
        if colSom == 0:
            A[:, i] = 1/N # Si somme = 0, alors la colonne entière devienne 1/N
        else:
            A[:, i] = alpha * A[:, i] + (1-alpha)/N # Sinon, chaque cellule devienne alpha * cellule + (1-alpha)/N
    return A

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

alpha = 0.85

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
    Q = matrice_stochastique(matrice_transposee(A))

    # Traitement par colonne
    P = matrice_transition_P(Q, alpha)

    # Dimension de la matrice
    d = P.shape[0]
    r = np.ones(d) / d # Normalisation

    while True:
        ancien_r = r
        r = np.dot(P, r)
        # On compare deux matrices selon la précision p
        if np.allclose(r, ancien_r, atol=p):
            return ancien_r

"""
1. On rappelle que le vecteur de score est solution du système 
$r = Pr$. En déduire un algorithme de calcul direct 
(c’est-à-dire de calcul exact et sans approximations successives)
du score $r$. 
Ecrire le pseudo-code correspondant à cet algorithme.
"""

web1 = [
    0, 0, 0, 0, 0,
    1, 0, 0, 0, 0,
    1, 0, 0, 1, 0,
    1, 1, 0, 0, 0,
    1, 1, 0, 0, 0
]
print("Matrice: \n", np.array(web1).reshape(int(np.sqrt(len(web1))), int(np.sqrt(len(web1)))))

a = matrice_transposee(web1)
print("Transposée: \n", a)

aa = matrice_stochastique(a)
print("Stochastique: \n", aa)

aaa = matrice_transition_P(aa, 0.85)
print("Facteur d'amortissement: \n", aaa)

def puissance_iteree_exacte(A, alpha=0.85):
    Q = matrice_stochastique(matrice_transposee(A))
    P = matrice_transition_P(Q, alpha)
    
    # Dimension de la matrice
    d = P.shape[0]

    # Génére une matrice d'identité dimension d
    # la diagonale à 1
    # [[1,0,0]
    #  [0,1,0]
    #  [0,0,1]]
    I = np.eye(d)
    print("Identitée: \n", I)

    matrice_gauss = P - I
    print(matrice_gauss)
    inconnues = []
    for inconnue in range(d):
        inconnues.append(matrice_gauss[inconnue][inconnue])
    
puissance_iteree_exacte(web1)
