#
import torch                # type: ignore
import random
from matplotlib import pyplot as plt
import numpy as np
#
import lib_classes as lc
import gen_expr as ge

#
LINSPACE_START: int = -20
LINSPACE_STOP: int = 20
LINSPACE_NUM: int = 10000

#
def calculate_linspaces_of_expr(expr: lc.MathExpr) -> tuple[np.ndarray, np.ndarray]:

    #
    linspace: np.ndarray = np.linspace(start=LINSPACE_START, stop=LINSPACE_STOP, num=LINSPACE_NUM)

    #
    expr.eval()

    #
    X: torch.Tensor = torch.unsqueeze(torch.from_numpy(linspace), dim = -1)

    #
    # print(f"X = {X}")
    # print(f"X.shape = {X.shape}")

    #
    Y: torch.Tensor = expr( X )

    #
    # print(f"Y = {Y}")
    # print(f"Y.shape = {Y.shape}")

    #
    return linspace, Y.detach().numpy()


#
def display_expr(expr: lc.MathExpr) -> None:

    #
    X: np.ndarray
    Y: np.ndarray
    X, Y = calculate_linspaces_of_expr(expr=expr)

    #
    plt.plot(X, Y)
    plt.yscale("symlog")
    plt.title(f"${expr.to_latex()}$")

    #
    manager = plt.get_current_fig_manager()

    # Qt backend
    if hasattr(manager.window, "showMaximized"):    # type: ignore
        manager.window.showMaximized()              # type: ignore

    # TkAgg backend
    elif hasattr(manager.window, "attributes"):     # type: ignore
        manager.window.attributes('-zoomed', True)  # type: ignore

    #
    else:
        print("Maximizing the plot window is not supported by the current backend.")

    #
    plt.show()



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
            display_expr(expr=generated_expr)

        except Exception as e:

            #
            print(f"Error : {e}")
