# Partie 3 : PageRank - version itérative, analyse

Pour le moment on pose $\alpha= 0,85$ et on considère le graphe de la Partie 1.

1. Analyser l’influence du critère d'arrêt dans l’algorithme de la puissance itérée.
2. Ajouter quelques hubs (pages qui ont beaucoup de liens sortant) et autorités (pages
qui ont beaucoup de liens entrant). Commenter l’impact sur les scores.
3. Essayez d'accroître le score de certaines pages. Expliquez votre méthode et validez-la
expérimentalement.
4. Faites varier le facteur d’amortissement $α$ pour analyser son influence. On rappelle que
$α ∈ [0, 1]$.