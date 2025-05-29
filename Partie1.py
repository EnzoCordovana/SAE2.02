import numpy as np
import math
"""
Fonction norme qui prend en entrée un vecteur X et qui calcule sa norme.

Arguments:
X   vecteur, doit être un tableau
"""
def norme(X):
    sum = 0
    # On ajoute le carré à la varibale
    for x in np.array(X):
        sum += x**2
    # On retourne la racine à la variable
    return np.sqrt(sum)


"""
Fonction puissance_iteree qui prend 

A       matrice, doit être un tableau
p       précision, doit être un entier   
"""
def puissance_iteree(A, p):
    A = np.array(A)
    
    # Taille de la matrice carré
    t = len(A)

    # Dimention de la matrice carré
    d = int(math.sqrt(t))

    # Vecteur X_O \in R^N
    vec_0 = np.array([])
    for _ in range(d):
        vec_0 = np.append(vec_0, np.random.randint(-14, 14))

    # Redimensionner la matrice en une matrice carrée
    # [a,b,c,d] => [[a,b],[c,d]]
    A = A.reshape((d, d))

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
    

A = [2,0,0,3]

B = [
    4, 1, 2,
    0, 2, 5,
    3, 1, 3
]

C = [
    4, -1, 2, 9,
    0, 2, -5, 23,
    -3, 1, 3, -21,
    21, 3, 1, 0
]

print(puissance_iteree(A, 1e-6))
print(puissance_iteree(B, 1e-6))
print(puissance_iteree(C, 1e-6))