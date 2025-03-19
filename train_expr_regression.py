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
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(torch.abs(pred) + 1), torch.log(torch.abs(actual) + 1)))


#
def training_expr(model_expr: lc.MathExpr, x_input: Tensor, y_goal: Tensor, nb_epoch: int = 20, learning_rate: float = 0.01, device: str = "cuda" if torch.cuda.is_available() else "cpu", goal_expr: Optional[lc.MathExpr] = None, dir_to_save_imgs: str = "", save_fig_each_epochs_modulo: int = 1, nb_batches: int = 100, nb_sub_epochs_per_batches: int = 10) -> None:

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
    derivated_goal = y_goal.roll(shifts=1)
    derivated_goal[0] = derivated_goal[1]
    derivated_goal = (- torch.abs(y_goal - derivated_goal)).squeeze()

    #
    topk_indices_goal = torch.topk( derivated_goal, k = 50 ).indices

    #
    optimizer: optim.Optimizer = optim.Adam(params=model_expr.parameters(), lr=learning_rate)

    #
    # loss_fn: nn.Module = nn.MSELoss()
    loss_fn: nn.Module = RMSLELoss()

    #
    model_expr = model_expr.to(device)

    #
    for epoch in range(nb_epoch):

        #
        if epoch % 200 > 100:

            #
            model_expr.eval()

            #
            y_learning: Tensor = learning_expr(x_input)
            #
            derivated_learning = y_learning.roll(shifts=1)
            derivated_learning[0] = derivated_learning[1]
            derivated_learning = (- torch.abs(y_learning - derivated_learning)).squeeze()

            #
            topk_indices_learning = torch.topk( derivated_learning, k = 50 ).indices

            #
            topk_indices = torch.cat( (topk_indices_goal, topk_indices_learning) )

            #
            x_masked_input = x_input[topk_indices].unsqueeze(-1)
            y_masked_goal = y_goal.squeeze()[topk_indices].unsqueeze(-1)

        #
        else:

            #
            x_masked_input = x_input
            y_masked_goal = y_goal

        #
        losses_epoch: list[float] = []

        # Subdivide x_masked_input and y_masked_goal into batches
        indices = torch.arange(x_masked_input.size(0))
        # indices = torch.randperm(indices)
        batch_size = x_masked_input.size(0) // nb_batches

        for _ in range(10): #iterate 10 times per batch
            for i in range(nb_batches):
                start = i * batch_size
                end = (i + 1) * batch_size if i < nb_batches - 1 else x_masked_input.size(0)
                batch_indices = indices[start:end]

                x_batch = x_masked_input[batch_indices].to(device)
                y_batch = y_masked_goal[batch_indices].to(device)

                #
                model_expr.train()

                #
                optimizer.zero_grad()
                model_expr.zero_grad()

                #
                output = model_expr(x_batch).to(device)

                #
                loss = loss_fn(output, y_batch)

                #
                loss.backward()

                #
                losses_epoch.append( loss.item() )

                #
                optimizer.step()

        #
        avg_loss_epoch: float = sum(losses_epoch) / len(losses_epoch)
        print(f"epoch {epoch} | loss = {avg_loss_epoch}")

        #
        if avg_loss_epoch < 0.003:

            #
            print(f"finished learning !\nLoss {avg_loss_epoch} < 0.003 !")

            #
            if goal_expr is not None and dir_to_save_imgs != "":

                #
                model_expr = model_expr.eval()

                #
                de.display_exprs(expr1=model_expr, expr2=goal_expr, title=f"frame {epoch}", save_to_file=f"{dir_to_save_imgs}epoch_{epoch:03}.png")

            #
            return

        #
        if goal_expr is not None and dir_to_save_imgs != "" and epoch % save_fig_each_epochs_modulo == 0:

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
        nb_epoch=1000,
        learning_rate=0.01,
        goal_expr=goal_expr,
        dir_to_save_imgs=f"imgs/test{len(os.listdir("imgs/"))}/"
    )

