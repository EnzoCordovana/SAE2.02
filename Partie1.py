import numpy as np

"""
Fonction norme qui prend en entrée un vecteur X et qui calcule sa norme.

Arguments:
X   vecteur, doit être un tableau
"""
def norme(X):
    sum = 0
    for x in np.array(X):
        sum += x**2
    return np.sqrt(sum)

vec = [1,2,3,4]

print(norme(vec))