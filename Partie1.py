import numpy as np

"""
Fonction norme qui prend en entrée un vecteur X et qui calcule sa norme.

Arguments:
X   vecteur, doit être un tableau
"""
def norme(X):
    sum = 0
    for x in np.array(X): # On ajoute le carré à la varibale
        sum += x**2
    return np.sqrt(sum) # On retourne la racine à la variable