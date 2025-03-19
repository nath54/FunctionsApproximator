#
from typing import Optional
#
import os
import random
import torch                    # type: ignore
from torch import Tensor        # type: ignore
import torch.nn as nn           # type: ignore
import torch.optim as optim     # type: ignore
import numpy as np              # type: ignore

#
import lib_classes as lc
import gen_expr as ge
import display_expr as de


#
def training_expr(model_expr: lc.MathExpr, x_input: Tensor, y_goal: Tensor, nb_epoch: int = 20, learning_rate: float = 0.01, device: str = "cuda" if torch.cuda.is_available() else "cpu", goal_expr: Optional[lc.MathExpr] = None, dir_to_save_imgs: str = "") -> None:

    #
    if goal_expr is not None and dir_to_save_imgs != "":

        #
        if not os.path.exists(dir_to_save_imgs):

            #
            os.makedirs(dir_to_save_imgs)

        #
        if not dir_to_save_imgs.endswith("/"):

            dir_to_save_imgs = dir_to_save_imgs + "/"

    #
    optimizer: optim.Optimizer = optim.Adam(params=model_expr.parameters(), lr=learning_rate)

    #
    loss_fn: nn.Module = nn.MSELoss()

    #
    model_expr = model_expr.to(device)

    #
    for epoch in range(nb_epoch):

        #
        model_expr.train()

        #
        optimizer.zero_grad()
        model_expr.zero_grad()

        #
        output = model_expr(x_input).to(device)

        #
        loss = loss_fn(output, y_goal)

        #
        loss.backward()

        #
        print(f"epoch {epoch} | loss = {loss.item()}")

        #
        optimizer.step()

        #
        if goal_expr is not None and dir_to_save_imgs != "":

            #
            model_expr = model_expr.eval()

            #
            de.display_exprs(expr1=model_expr, expr2=goal_expr, title=f"frame {epoch}", save_to_file=f"{dir_to_save_imgs}epoch_{epoch:03}.png")




#
if __name__ == "__main__":

    #
    goal_expr: lc.MathExpr = ge.gen_random_math_expr(nb_actions_to_generate=random.randint(10, 50))

    #
    learning_expr: lc.MathExpr = goal_expr.duplicate()

    #
    print(f"Goal expr : {goal_expr.to_latex()}")
    print(f"Learning expr : {learning_expr.to_latex()}")

    #
    x_input_np: np.ndarray
    y_goal_np: np.ndarray
    x_input_np, y_goal_np = de.calculate_linspaces_of_expr(expr=goal_expr)

    #
    x_input: Tensor = torch.from_numpy(x_input_np)
    y_goal: Tensor = torch.from_numpy(y_goal_np)

    #
    training_expr(
        model_expr=learning_expr,
        x_input=x_input,
        y_goal=y_goal,
        nb_epoch=100,
        learning_rate=0.5,
        goal_expr=goal_expr,
        dir_to_save_imgs=f"imgs/test{len(os.listdir("imgs/"))}/"
    )

