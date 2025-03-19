#
from typing import Optional
#
import torch                # type: ignore
from matplotlib import pyplot as plt
import numpy as np
#
import lib_classes as lc


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
def display_expr(expr: lc.MathExpr, nb_curves: int = 10) -> None:

    # Close all previous figures
    plt.close('all')

    # Create a fresh figure
    plt.figure()

    # Clear the plot just in case
    plt.clf()
    plt.cla()

    #
    X: np.ndarray
    Y: np.ndarray

    #
    for i in range(nb_curves):

        #
        X, Y = calculate_linspaces_of_expr(expr=expr.duplicate())
        plt.plot(X, Y)

    #
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
def display_exprs(expr1: lc.MathExpr, expr2: lc.MathExpr, title: str = "expressions plot", maximized: bool = True, save_to_file: Optional[str] = None) -> None:

    # Close all previous figures
    plt.close('all')

    # Create a fresh figure
    plt.figure()

    # Clear the plot just in case
    plt.clf()
    plt.cla()

    #
    X1: np.ndarray
    Y1: np.ndarray
    X2: np.ndarray
    Y2: np.ndarray

    #
    X1, Y1 = calculate_linspaces_of_expr(expr=expr1)
    X2, Y2 = calculate_linspaces_of_expr(expr=expr2)

    #
    plt.plot(X2, Y2, "r")
    plt.plot(X1, Y1, "b")
    #
    plt.yscale("symlog")
    plt.title(title)

    #
    if save_to_file is not None:

        #
        plt.savefig(save_to_file)

        #
        return


    #
    if maximized:

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
