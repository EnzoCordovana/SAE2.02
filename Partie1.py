import numpy as np
import math

def matrice_transposee(A):
    """
    Fonction matrice_transposee qui prend en entrée 
    une matrice A et qui renvoie sa transposée.

    Arguments:
    A   matrice, doit être un tableau
    """
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
    """
    Fonction nomatrice_stochastiquerme qui prend en entrée 
    une matrice C et qui renvoie sa stochastique.

    Arguments:
    C   matrice, doit être un tableau
    """
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


# Fonction vue en TP
def puissance_iteree(C, p):
    """
    Fonction puissance_iteree qui prend en entrée 
    une matrice A, une précision p et qui retourne 
    la valeur propre ainsi que le vecteur propre.
    
    A       matrice, doit être un tableau.
    p       précision, doit être un entier.
    """
    C = np.array(A)

    # Dimension de la matrice carré
    d = int(math.sqrt(len(C)))

    # Vecteur X_O \in R^N
    vec_0 = np.array([])
    for _ in range(d):
        vec_0 = np.append(vec_0, np.random.randint(-14, 14))

    # Redimensionner la matrice en une matrice carrée
    # [a,b,c,d] => [[a,b],[c,d]]
    C = C.reshape((d, d))

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
# Graphe de la partie 1 pointant i vers j
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

# Fonction pour la SAE
def puissance_iteree_v2(C, p):
    """
    Fonction puissance_iteree qui prend en entrée 
    une matrice A, une précision p et qui retourne 
    un tableau de vecteur propre.
    
    Arguments:
    C       matrice, doit être un tableau
    p       précision, doit être un entier   
    """

    C = np.array(C)

    # Transposé de la matrice
    C = matrice_transposee(C)

    # Calcule de la matrice Q
    Q = matrice_stochastique(C)

    # Nombre de page
    N = Q.shape[0]

    # Vecteur initial
    r = np.ones(N) / N

    while True:
        ancien_r = r.copy()
        r = np.dot(Q, r)
        # On compare deux matrices selon la précision p
        if np.allclose(r, ancien_r, atol=p):
            return ancien_r, Q

precision = 1e-6

# Calcul des scores des pages de la matrice web
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


# On crée une liste de tuples (index_page, score)
page_rank = list(enumerate(r, start=1))
page_rank.sort(key=lambda tup: tup[1], reverse=True)

listeLiensEntrants = [5,1,2,2,3,3,1,3,1,5,1,2,2,3]
listeLiensSortants = [5,3,2,2,1,3,2,1,2,5,3,2,2,1]

print("PageRank :\t\tEntrants\tSortants")
for page, rank in page_rank:
    print(f"Page {page}\t: {rank:.4f}\t" + str(listeLiensEntrants[page-1]) + "\t\t" + str(listeLiensSortants[page-1]))