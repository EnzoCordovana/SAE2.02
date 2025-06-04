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

webWithHub = [
    0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,   # Page 1 [Hub] -> {2, 3, 4, 5, 6}
    1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,   #Page 10 [Hub] -> {6, 9, 11, 12, 13, 14}
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
]

webWithAuthority = [
    0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   #Page 6 [Authority] -> {8}
    1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   #Page 14 [Authority] -> {}
]

webWithHubAndAuthority = [
    0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,   # Page 1 [Authority] -> {2, 6}
    1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,   #Page 6 [Hub] -> {1, 7, 8, 9, 10}
    1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   #Page 8 [Authority] -> {}
    0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
    0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0,   #Page 14 [Hub] -> {2. 5, 6, 10, 11, 13}
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
    iterations = 0

    while True:
        ancien_r = r
        r = np.dot(Q, r)
        # On incrémente le nombre d'itérations à chaque boucle
        iterations += 1
        # On compare deux matrices selon la précision p
        if np.allclose(r, ancien_r, atol=p):
            print(f"Convergence atteinte en {iterations} itérations avec p = {p}")
            return ancien_r

r = puissance_iteree_v2(web, 1e-6, alpha)

#Partie 3.2 - Web avec Hub, Authority et Hub+Authority
# On applique la fonction puissance_iteree_v2 sur les différents graphes
rH = puissance_iteree_v2(webWithHub, 1e-6, 0.85)
rA = puissance_iteree_v2(webWithAuthority, 1e-6, 0.85)
rHA = puissance_iteree_v2(webWithHubAndAuthority, 1e-6, 0.85)

# On crée une liste de tuples (index_page, score)
page_rank = list(enumerate(r, start=1))
page_rank2 = list(enumerate(rH, start=1))
page_rank3 = list(enumerate(rA, start=1))
page_rank4 = list(enumerate(rHA, start=1))



print("PageRank :")
precisions = [1e-2, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]

for page, rank in page_rank:
    print(f"Page {page}\t: {rank:.4f}")

print("\n")
print("===========================")
print("\nPageRank avec un Hub :")

for page, rank in page_rank2:
    print(f"Page {page}\t: {rank:.4f}")   


print("\n")
print("===========================")
print("\nPageRank avec une Authority :")

for page, rank in page_rank3:
    print(f"Page {page}\t: {rank:.4f}")


print("\n")
print("===========================")
print("\nPageRank avec un Hub et une Authority :")

for page, rank in page_rank4:
    print(f"Page {page}\t: {rank:.4f}")


for p in precisions:
    print(f"\nCalcul avec précision {p}:")
    r = puissance_iteree_v2(web, p, 0.85)
