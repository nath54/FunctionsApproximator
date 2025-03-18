

Base d'une expression:

- base_expr = Somme([])



Termes que l'on peut ajouter à un élément:


- MathExpr
    - Terme Constant:       a
    - MathExprEltList
        - Terme Somme:          Somme([])
        - Terme Produit:        Produit([])
    - MathExprComposed
        - Terme Polynomial:     a * X ^ b
        - Terme Exponentiel:    a * b ^ X
        - Terme Logarithmique:  a * log_base(X)
        - Terme Inverse:        a / (X)
        - Terme Sinusoïdal:     a * sin( b * X + c )


X -> peux être composé par un autre terme



Somme():

    Somme vide = 0

    -> ajouter un terme
    -> supprimer un terme


Produit():

    Produit vide = 1

    -> ajouter un terme
    -> supprimer un terme


MathExprComposed:

    Si non composée
        -> Ajouter une composée

    Si composée:
        -> Supprimer la composée







