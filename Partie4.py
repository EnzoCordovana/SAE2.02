import numpy as np

"""
1. Utiliser le logiciel d'exploration de site web pour
construire 3 matrices d'ordre [10, 30].
Appliquer l'algorithme de PageRank
"""

# nextjs 10*10 : nextjs.org
matrice1 = [
    1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,
    1,1,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,0,0,0,
    1,1,0,1,0,0,0,1,1,1,
    1,1,0,0,0,0,0,0,0,0,
]

# github 10*10 : github.com
matrice1i = [
    1,0,1,1,1,1,1,1,1,1,
    1,0,0,0,0,0,0,0,0,0,
    1,0,1,1,1,1,1,1,1,1,
    1,0,1,1,1,1,1,1,1,1,
    1,0,1,1,1,1,1,1,1,1,
    1,0,1,1,1,1,1,1,1,1,
    1,0,1,1,1,1,1,1,1,1,
    1,0,1,1,1,1,1,1,1,1,
    1,0,1,1,1,1,1,1,1,1,
    1,0,1,1,1,1,1,1,1,1,
]

# astro 20*20 : astro.build
matrice2 = [
    1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,
    1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
]

# svelte 30*30 : svelte.dev
matrice3 = [
    1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
]

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

matrices = [["Next js: ", matrice1],["Astro :", matrice2],["Svelte: ", matrice3]]

# Affichage du score de chaque page de chaque matrice
for matrice_info in matrices:
    nom, matrice = matrice_info
    page_rank = list(enumerate(puissance_iteree_v2(matrice, 1e-6, alpha), start=1))
    print(nom)
    for page, rank in page_rank:
        print(f"Page {page}\t: {rank:.4f}")
    print("==========================")

"""
2. 
"""

# Matrice de la ville
M = np.load("Data/Nice/Nice_France_Matrice.npy")
# Liste des identifiants des noeuds
N = np.load("Data/Nice/Nice_France_Id_Noeud.npy")

# print('M: ', M,'N: ', N)