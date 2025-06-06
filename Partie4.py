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

matrices = [
    [
        "Next js: ",
        matrice1
    ],
    [
        "Astro :",
        matrice2
    ],
    [
        "Svelte: ", 
        matrice3
    ]
]

alpha = 0.85

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

def matrice_transition_P(A, alpha):
    for i in range(len(A)):
        colSom = np.sum(A[:, i]) # Stockage de la somme d'une colonne
        if colSom == 0:
            A[:, i] = [1/len(A)] # Si somme = 0, alors la colonne entière devienne 1/N
        else:
            A[:, i] = alpha * A[:, i] + (1-alpha)/len(A) # Sinon, chaque cellule devienne alpha * cellule + (1-alpha)/N
    return A

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
            return ancien_r, Q

r, Q = puissance_iteree_v2(matrice1i, 1e-6, alpha)
for page in range(len(r)):
    print(f"Page {page+1:<3}: {r[page]:.4f}")

matrices = [["Next js: ", matrice1],["Astro :", matrice2],["Svelte: ", matrice3]]

# Affichage du score de chaque page de chaque matrice
#for matrice_info in matrices:
#    nom, matrice = matrice_info
#    r, Q = list(enumerate(puissance_iteree_v2(matrice, 1e-6, alpha), start=1))
#    print(nom)
#    for page, rank in r:
#        print(f"Page {page}\t: {rank:.4f}")
#    print("==========================")

"""
2. 
"""

## Nice
# Matrice de la ville
M_nice = np.load("Data/Nice/Nice_France_Matrice.npy")
# Liste des identifiants des noeuds
N_nice = np.load("Data/Nice/Nice_France_Id_Noeud.npy")

## Aix
# Matrice de ville
M_aix = np.load("Data/Aix/413_avenue_Gaston_Berger_Aix_en_Provence_France_Matrice.npy")
# Liste des identifiants des noeuds
N_aix = np.load("Data/Aix/413_avenue_Gaston_Berger_Aix_en_Provence_France_Id_Noeud.npy")

# Fonction pour afficher le noeud sur la map
def url(noeud):
    return "https://www.openstreetmap.org/node/" + str(noeud)


def afficher_top10(r, noeuds, nom_ville):
    """
    Affiche les 10 meilleurs nœuds avec leurs scores PageRank
    """
    print(f"\n=== TOP 10 - {nom_ville} ===")
    print("Rang\tScore PageRank\t\tID Nœud\t\tURL OpenStreetMap")
    print("-" * 80)
    
    # Créer une liste de tuples (score, index, noeud_id)
    scores_avec_index = [(r[i], i, noeuds[i]) for i in range(len(r))]
    
    # Trier par score décroissant
    scores_tries = sorted(scores_avec_index, key=lambda x: x[0], reverse=True)
    
    # Afficher les 10 premiers
    for rang in range(min(10, len(scores_tries))):
        score, index, noeud_id = scores_tries[rang]
        print(f"{rang+1}\t{score:.6f}\t\t{noeud_id}\t\t{url(noeud_id)}")

villes = [
    [
        "Nice: ",
        M_nice.T,
        N_nice,
    ],
    [
        "Aix: ",
        M_aix,
        N_aix
    ]
]

print(f"Nice: Matrice {M_nice.shape}, {len(N_nice)} nœuds")
print(f"Aix: Matrice {M_aix.shape}, {len(N_aix)} nœuds")

# Calcul PageRank pour chaque ville
for nom_ville, matrice, noeuds in villes:
    print(f"\n{'='*50}")
    print(f"Calcul PageRank pour {nom_ville}")
    print(f"{'='*50}")
    
    # Calcul des scores PageRank
    r, P = puissance_iteree_v2(matrice, 1e-6, alpha)
        
    print(f"Matrice: {matrice.shape}")
    print(f"Nombre de nœuds: {len(noeuds)}")
    print(f"Score moyen: {np.mean(r):.6f}")
    print(f"Score max: {np.max(r):.6f}")
    print(f"Score min: {np.min(r):.6f}")
    
    # Afficher le top 10
    afficher_top10(r, noeuds, nom_ville)