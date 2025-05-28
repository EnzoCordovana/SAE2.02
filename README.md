# SAE2.02
Exploration algorithmique d'un problème

*PageRank & applications*

## Consignes

Pour la partie implémentation, vous utiliserez `Python` ou `Sagemath` avec les bibliothèques
nécessaires. Pour le rapport, afin de présenter du texte, du code ainsi que le rendu du code
on utilisera `jupyter-notebook`.

Le dossier rendant compte du projet est à rendre sur la page Ametice de la SAE au format
HTML et au format ipynb. La date limite de rendu est le **9 juin, 12h00**. Un seul étudiant par
équipe déposera le dossier. Outre le code et l’analyse des résultats, le dossier présentera de
façon exhaustive la participation de chaque étudiant au projet (qui a fait quoi) ainsi qu’un
pourcentage d’implication dans le projet (qui a fait combien). Chaque membre du groupe
devra maˆıtriser l’intégralité de ce qui est rendu.

L’oral aura lieu le 11 juin et sera une séance de questions sur le rapport d’une durée de 15
minutes par groupe.

**Le dossier devra contenir les réponses aux 6 parties décrites ci-dessous.** Il est bien évidemment
attendu que vous compreniez le fonctionnement de l’algorithme `PageRank` mais dans le do-
cument à rendre il n’est pas nécessaire de re-détailler le fonctionnement de cet algorithme
présenté dans la section ci-dessous.

Bien évidemment la totalité du rendu aura été écrit par l’un des membres du groupes. Le
recours à toute aide extérieure est interdit.

## Principe de PageRank

La recherche d’informations pertinentes sur le Web est un des problèmes les plus cruciaux
pour l’utilisation de ce dernier. Il y a des enjeux économiques colossaux. Le leader actuel de
ce marché, Google, utilise pour d´eterminer la pertinence des références fournies, un certain
nombre d’algorithmes dont certains sont des secrets industriels jalousement gardés, mais
d’autres sont publics.

On va s’intéresser à (une partie de) l’algorithme `PageRank`. L’idée de base utilisée par les
moteurs de recherche pour classer les pages du web par ordre de pertinence d´ecroissante
consiste à considérer que plus une page est la cible de liens venant d’autres pages, c’est-
à-dire plus il y a de pages qui pointent vers elle, plus elle a de chances d’ˆetre fiable et
intéressante, et réciproquement. Il s’agit donc de quantifier cette idée, c’est-à-dire d’attri-
buer un rang numérique ou score de pertinence à chaque page.

On numérote les pages du web de $1$ a $N$ ($N$ est très grand, de l’ordre de $10^{12}$) et on note $r_i > 0$ le score de la page $i$. **C’est ce score que l’on cherche à calculer.**

On considère le Web comme un graphe orienté (vu en détails dans le TP de R2.07 dédiè à la
SAE) dont les sommets sont les pages numérotées de $1$ à $N$. Il y a une arˆete de $i$ vers $j$ s’il y a sur la page $i$ un lien hypertexte vers la page $j$ (on dit que $i$ pointe vers $j$). On définit la matrice $C = (ci,j)_{1≤i,j≤N}$ de taille $N × N$ par
$$
c_{i,j} = \begin{cases}
1 & \text{si } j \text{ pointe vers } i \text{ et } i \neq j, \\
0 & \text{sinon.}
\end{cases}
$$

C’est donc la matrice transposée de la matrice d’adjacence du graphe du web. </br>
L’algorithme `PageRank` part du principe que toutes les pages n’ont pas le même poids :
- Plus la page $i$ a un score élevé plus les pages $j$ vers lesquelles la page $i$ pointe auront
un score important;
- Plus il y a de liens différents sur la page $j$ et moins on attribue d’importance à chaque
lien.

Ainsi, en notant $N_j$ le nombre de liens présents sur la page $j$, on aboutit au système suivant

$$
r_i = \sum_{j=1}^{N} \frac{c_{i,j}}{N_j} \times r_j, \quad \text{pour tout } i \in \{1, \ldots, N\}.
$$

On pourra remarquer que $r_i = \sum_{j=1}^{N} \frac{c_{i,j}}{N_j}$ </br>
Finalement en adoptant les notations matricielles suivantes

$$
r = \begin{pmatrix} r_1 \\ \vdots \\ r_N \end{pmatrix} \in \mathbb{R}^N \quad \text{et} \quad Q = \left( q_{i,j} \right)_{1 \leq i,j \leq N} \quad \text{avec} \quad q_{i,j} = \begin{cases}
\frac{c_{i,j}}{N_j} & \text{si } N_j \neq 0, \\
0 & \text{sinon.}
\end{cases}
$$

on aboutit à l’équation suivante : le vecteur $r$ des scores est solution de r = Qr. </br>
L’algorithme `PageRank` peut être utilisé pour des graphes qui ne sont pas associés au graphe
du Web, dans ce cas ses résultats nécessitent une interprétation précise.