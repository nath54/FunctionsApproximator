#
import random
#
import lib_classes as lc
import gen_expr as ge
import display_expr as de



#
if __name__ == "__main__":

    #
    for i in range(100):

        try:

            #
            generated_expr: lc.MathExpr = ge.gen_random_math_expr(nb_actions_to_generate=random.randint(10, 50))

            #
            print(f"Generated expr : {generated_expr.to_latex()}")

            #
            de.display_expr(expr=generated_expr)

        except Exception as e:

            #
            print(f"Error : {e}")
