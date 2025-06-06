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

def matrice_transition_P(A, alpha):
    for i in range(len(A)):
        colSom = np.sum(A[:, i]) # Stockage de la somme d'une colonne
        if colSom == 0:
            A[:, i] = [1/len(A)] # Si somme = 0, alors la colonne entière devienne 1/N
        else:
            A[:, i] = alpha * A[:, i] + (1-alpha)/len(A) # Sinon, chaque cellule devienne alpha * cellule + (1-alpha)/N
    return A

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
    matrice_transition_P(Q, alpha)

    d = Q.shape[0]
    r = np.ones(d) / d # Normalisation

    while True:
        ancien_r = r
        r = np.dot(Q, r)
        # On compare deux matrices selon la précision p
        if np.allclose(r, ancien_r, atol=p):
            return ancien_r

def preparer_matrice_google(A, alpha):
    """
    Fonction preparer_matrice_google qui prend en entrée 
    une matrice A et un alpha et qui retourne la matrice 
    de transition P.
    
    A       matrice, doit être un tableau
    alpha   coefficient de damping, doit être un float
    """
    A = np.array(A)

    # Si c'est un vecteur aplati, le transformer en matrice carrée
    if A.ndim == 1:
        d = int(np.sqrt(len(A)))
        A = A.reshape((d, d))

    # On transpose la matrice et on la rend stochastique
    P = matrice_stochastique(matrice_transposee(A))

    #Traitement de la matrice P
    # On applique le principe de Google PageRank
    n = P.shape[0]  # Nombre de pages
    for i in range(n):
        if np.sum(P[:, i]) == 0:
            P[:, i] = np.ones(n) / n
        else:
            P[:, i] = alpha * P[:, i] + (1 - alpha) / n

    return P

def puissance_direct(A, alpha):
    """
    PARTIE 5.1 et 5.2 : Algorithme de calcul direct du PageRank
    
    Pseudo-code:
    1. Construire la matrice Google G = alpha * P + (1-alpha)/n * ones
    2. Construire la matrice (I - G)
    3. Remplacer une ligne par la contrainte de normalisation (somme = 1)
    4. Résoudre le système linéaire (I - G_modifié) × r = b
    5. Retourner r
    
    A: matrice d'adjacence
    alpha: facteur d'amortissement
    verbose: affichage des détails
    
    Returns: (vecteur PageRank, temps d'exécution)
    """

    start_time = time.time()

    # Préparation de la matrice Google
    P = preparer_matrice_google(A, alpha)

    # Création de la matrice identité
    I = np.eye(P.shape[0])

    # Création de la matrice (I - P)
    G = I - P

    # Remplacement de la dernière ligne par la contrainte de normalisation
    G[-1, :] = 1

    # Résolution du système linéaire
    b = np.zeros(G.shape[0])
    b[-1] = 1  # Contrainte de normalisation

    r = np.linalg.solve(G, b)

    end_time = time.time()
    execution_time = end_time - start_time

    return r, execution_time



r = puissance_iteree_v2(web, 1e-6, alpha)

# On crée une liste de tuples (index_page, score)
page_rank = list(enumerate(r, start=1))

print("PageRank :")
for page, rank in page_rank:
    print(f"Page {page}\t: {rank:.4f}")
print("\r")
print("r = Pr approximativement ?")
A = np.array(web)
Q = matrice_stochastique(matrice_transposee(A))
Pr = r.dot(matrice_transition_P(Q, alpha))
res= []
tabRes = abs(np.subtract(Pr, r))
for i in tabRes:
    if i < 0.85:
        res.append(True)
    else:
        res.append(False)
print(res)





"""
1. L'algorithme de calcul direct du score r par rapport au système r=Pr où P est la matrice de transition est (I - P)r = 0.Pour résoudre ce système, on peut le réécrire comme :

r = Pr
r - Pr = 0
(I - P)r = 0

Cependant, ce système homogène a une solution triviale (r = 0). En réalité, on cherche le vecteur propre principal de P, c'est-à-dire la solution de (I - P)r = 0 avec la contrainte que la somme des composantes de r égale 1.

"""