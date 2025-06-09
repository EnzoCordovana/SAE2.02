import numpy as np
import math
import matplotlib.pyplot as plt
import time

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

#3.2A - Ajout de Hub seulement
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

#3.2B - Ajout d'Authoritié seulement
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

#3.2C - Ajout des Hubs et des Authorités
webWithHubAndAuthority = [
    0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,   # Page 1 [Authority] -> {2, 6}
    1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1,   #Page 6 [Hub] -> {1, 7, 8, 9, 10}
    1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,   #Page 8 [Authority] -> {}
    0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
    1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1,
    1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0,   #Page 14 [Hub] -> {10, 11, 13}
]


# Partie 3.3 - Augementation du score des pages X
# Matrice de boost pour les pages X en ajoutant des liens entrants vers une page ciblée
webPageXScoreBoost = [
    0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,   #Page 1
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   #Page 2 -> Page 1
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   #Page 3 -> Page 1
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   #Page 4 -> Page 1
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   #Page 5 -> Page 1
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,   #Page 6 -> Page 1, 10
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,   #Page 7 -> Page 1
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,   #Page 8 -> Page 1
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,   #Page 9 -> Page 10, 1
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   #Page 10 > Page 1
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,   #Page 11 -> Page 10, 1
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,   #Page 12 -> Page 10, 1
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,   #Page 13 -> Page 10, 1
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,   #Page 14 -> Page 10, 1
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

def puissance_iteree_facteur_temps(A, p=1e-6, alpha=0.85):
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
    start_time = time.time()

    while True:
        ancien_r = r
        r = np.dot(Q, r)
        # On incrémente le nombre d'itérations à chaque boucle
        iterations += 1
        elapsed_time = time.time() - start_time
        # On compare deux matrices selon la précision p
        if np.allclose(r, ancien_r, atol=p):
            print(f"Convergence atteinte en {iterations} itérations avec p = {p} et alpha = {alpha}. Temps écoulé : {elapsed_time:.4f} secondes.")
            return ancien_r




# On applique la fonction puissance_iteree_facteur_temps sur les différents graphes
r = puissance_iteree_facteur_temps(web, 1e-6, alpha)
rH = puissance_iteree_facteur_temps(webWithHub, 1e-6, 0.85)
rA = puissance_iteree_facteur_temps(webWithAuthority, 1e-6, 0.85)
rHA = puissance_iteree_facteur_temps(webWithHubAndAuthority, 1e-6, 0.85)
rB = puissance_iteree_facteur_temps(webPageXScoreBoost, 1e-6, 0.85)

# On crée une liste de tuples (index_page, score)
page_rank = list(enumerate(r, start=1))
page_rank_Hub = list(enumerate(rH, start=1))
page_rank_Auth = list(enumerate(rA, start=1))
page_rank_HuAu = list(enumerate(rHA, start=1))
page_rank_BoostedPage = list(enumerate(rB, start=1))


print("PageRank :")
for page, rank in page_rank:
    print(f"Page {page}\t: {rank:.4f}")

# On extrait le premier rangs pour le graphique
rankWebNormal = []
for i in page_rank:
    rankWebNormal.append(i[1])


print(f"\n" + "=" * 60) #Séparateur
print("\nPageRank avec un Hub :")

for page, rank in page_rank_Hub:
    print(f"Page {page}\t: {rank:.4f}")   

rankHub = []
for i in page_rank_Hub:
    rankHub.append(i[1])

print(f"\n" + "=" * 60)
print("\nPageRank avec une Authority :")

for page, rank in page_rank_Auth:
    print(f"Page {page}\t: {rank:.4f}")

rankAuth = []
for i in page_rank_Auth:
    rankAuth.append(i[1])

print(f"\n" + "=" * 60)
print("\nPageRank avec un Hub et une Authority (Version1) :")

for page, rank in page_rank_HuAu:
    print(f"Page {page}\t: {rank:.4f}")

# On extrait le second rangs pour le graphique
rankWebHuAu = []
for i in page_rank_HuAu:
    rankWebHuAu.append(i[1])


print(f"\n" + "=" * 60)
print("\nPageRank avec 2 pages boosté :")

# On affiche le PageRank avec les pages boostées
for page, rank in page_rank_BoostedPage:
    print(f"Page {page}\t: {rank:.4f}")






# On effectue les tests avec différentes précisions
print(f"\n" + "=" * 60)
print("\nNombre d'itérations par présicion :")

# Liste des précisions pour les tests
precisions = [1e-1, 1e-3, 1e-6, 1e-9]

for p in precisions:
    print(f"\nCalcul avec précision {p}:")
    rP = puissance_iteree_facteur_temps(web, p, 0.85)
    page_rank_Precs = list(enumerate(rP, start=1))
    for page, rank in page_rank_Precs:
        print(f"Page {page}\t: {rank:.4f}")
   



# On effectue les tests avec différents alphas
print(f"\n" + "=" * 60)
print("\nNombre d'itérations par alpha :")

# Liste des alphas pour les tests
alphas = [0.1, 0.9, 0.85, 0.99]

for a in alphas:
    print(f"\nCalcul avec alpha {a}:")
    rAlpha = puissance_iteree_facteur_temps(web, 1e-6, a)
    page_rank_Alphas = list(enumerate(rAlpha, start=1))
    for page, rank in page_rank_Alphas:
        print(f"Page {page}\t: {rank:.4f}")




# Graphiques des analyses

def creerListePages(n):
    listePages = []
    for i in range(n):
        listePages.append("Page " + str(i+1))
    return listePages

qtePages = 14

listePages = creerListePages(qtePages)

matriceSansHubEtAut = rankWebNormal
matriceAvecHubEtAut = rankWebHuAu
matriceAvecHub = rankHub
matriceAvecAut = rankAuth

plt.bar(listePages, matriceSansHubEtAut, align='edge',width= 0.5, label="Ranks sans Hubs et Autorités")
plt.bar(listePages, matriceAvecHubEtAut, color='red', width=0.5, label="Ranks avec Hubs et Autorités")

plt.title("Impact des scores avec Hubs et Autorités")
plt.ylabel("Rank [0-1]")
plt.legend()
plt.show()


plt.bar(listePages, matriceAvecHub, align='center', color= 'blue', width=0.75, label="Ranks avec seulement des Hubs")
plt.bar(listePages, matriceAvecAut, color='green', width=0.5, label="Ranks avec seuelement des Autorités")
plt.bar(listePages, matriceSansHubEtAut, align='center', color='red',width= 0.25, label="Ranks sans Hubs et Autorités")

plt.title("Différence des scores avec seulemnt Hubs ou Autorités")
plt.ylabel("Rank [0-1]")
plt.legend()
plt.show()




"""
(1)Pour analyser le critère d'arrêt, on peut observer le nombre d'itérations nécessaires pour atteindre la convergence par rapport à la précision donnée. On doit d'abord ajouter un compteur d'itérations dans la fonction `puissance_iteree_facteur_temps` et afficher ce compteur à la fin de chaque exécution et analyser l'impact de la précision sur le résultat, on peut exécuter la fonction `puissance_iteree_facteur_temps` avec différentes valeurs de précision et observer comment les scores des pages changent. On observe que plus la précision est élevée, plus le nombre d'itérations augmente, ce qui peut ralentir le calcul. Cependant, les scores finaux convergent vers des valeurs stables, indiquant que la précision est atteinte.

(2)L'impact des scores des pages en ajoutant des Hubs et des Autoritès est significatif. Les Hubs et les Autorités influencent la manière dont les scores sont distribués entre les pages. Les Hubs, qui sont des pages avec de nombreux liens sortants, peuvent augmenter le score des pages qu'ils pointent, tandis que les Autorités, qui sont des pages avec de nombreux liens entrants, peuvent également recevoir un score plus élevé. En comparant les scores avec et sans Hubs et Autorités, on peut voir que certaines pages gagnent en importance dans le réseau, ce qui reflète leur rôle dans la structure du web.

(3)Afin d'augmenter le score des pages X, on peut ajouter des liens entrants vers ces pages dans la matrice d'adjacence. Cela peut être réalisé en modifiant la matrice d'adjacence pour inclure des liens supplémentaires vers les pages ciblées. Par exemple, si l'on souhaite augmenter le score de la page 1, on peut ajouter des liens entrants depuis les pages 2 à 5 vers la page 1. Cela augmentera le score de la page 1 dans le calcul du PageRank, car elle recevra plus de "votes" de la part des autres pages.

(4)On observe qu'apres la varition de l'alpha, les scores des pages changent en fonction de la valeur de alpha. Un alpha plus élevé (proche de 1) donne plus de poids aux liens existants, tandis qu'un alpha plus bas (proche de 0) favorise l'exploration aléatoire des pages. Cela peut influencer la manière dont les scores sont distribués entre les pages, en favorisant certaines pages qui ont des liens sortants importants ou en redistribuant le score de manière plus uniforme.



"""
