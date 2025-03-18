#
from typing import Callable
#
import random
#
import lib_classes as lc



#
def gen_random_math_expr_atom() -> lc.MathExpr:

    #
    constructor: Callable = random.choice( lc.MATH_EXPRS_VALUES )

    #
    return constructor()


#
def apply_action(base_expr: lc.MathExpr, action: tuple[str, list[lc.MathExpr]]) -> lc.MathExpr:

    #
    new_expr: lc.MathExpr

    #
    if action[0] == "add_element":

        #
        new_expr = gen_random_math_expr_atom()

        #
        action[1][-1].elts.append( new_expr )

    #
    elif action[0] == "add_compose":

        #
        new_expr = gen_random_math_expr_atom()

        #
        action[1][-1].composed = new_expr

    #
    elif action[0] == "remove_compose":

        #
        pass

    #
    elif action[0].startswith("remove_element"):

        #
        pass

    #
    return base_expr


#
def rec_cleaning(expr: lc.MathExpr) -> lc.MathExpr:

    #
    if isinstance(expr, lc.MathExprEltList):

        #
        for i in range(len(expr.elts)):

            #
            expr.elts[i] = rec_cleaning(expr=expr.elts[i])

    #
    elif isinstance(expr, lc.MathExpr_Composed) and expr.composed is not None:

        #
        expr.composed = rec_cleaning(expr=expr.composed)

    #
    if isinstance(expr, lc.MathExpr_InverseTerm):

        #
        if isinstance(expr.composed, lc.MathExpr_Const) and abs(expr.composed.const.data.item()) < 1e-6:

            #
            expr.composed.const.data += 1.0

    #
    if isinstance(expr, lc.MathExprEltList_Sum) and len(expr.elts) == 0:

        #
        const_value: lc.MathExpr_Const = lc.MathExpr_Const(parent=expr)
        const_value.const.data = lc.Tensor([1])

        #
        expr.elts.append( const_value )

    #
    return expr


#
def gen_random_math_expr(nb_actions_to_generate: int) -> lc.MathExpr:

    #
    base_expr: lc.MathExprEltList_Sum = lc.MathExprEltList_Sum()

    #
    for j in range(nb_actions_to_generate):

        #
        actions: list[tuple[str, list[lc.MathExpr]]] = base_expr.actions(history=[])

        #
        base_expr = apply_action(base_expr=base_expr, action=random.choice(actions))

    # Nettoyage: enlever toutes les divisions par zéro
    base_expr = rec_cleaning(expr=base_expr)

    #
    return base_expr
