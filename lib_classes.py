#
from typing import Optional, Callable
#
import torch                    # type: ignore
from torch import nn            # type: ignore
from torch import Tensor        # type: ignore



###############################################################################
#################################### UTILS ####################################
###############################################################################


#
def log(X: Tensor, base: Tensor) -> Tensor:

    #
    return torch.log(X) / torch.log(base)



#
def bd(value: Tensor | nn.Parameter) -> str:

    #
    if isinstance(value, nn.Parameter):

        #
        return f"{value.data.item(): .2f}"

    #
    else:

        #
        return f"{value.item(): .2f}"




###############################################################################
################################## MATH EXPR ##################################
###############################################################################

#
class MathExpr(nn.Module):

    #
    def __init__(self, parent: Optional["MathExpr"] = None) -> None:

        #
        super().__init__()

        #
        self.parent: Optional[MathExpr] = parent

    #
    def duplicate(self, parent: Optional["MathExpr"] = None) -> "MathExpr":

        #
        return MathExpr(parent=parent)

    #
    def actions(self, history: list["MathExpr"]) -> list[tuple[str, list["MathExpr"]]]:

        #
        return []

    #
    def get_self_parameters(self) -> list[nn.Parameter]:

        #
        return []

    #
    def get_compose_parameters(self) -> list[nn.Parameter]:

        #
        return []

    #
    def parameters(self) -> list[nn.Parameter]:

        #
        return self.get_self_parameters() + self.get_compose_parameters()

    #
    def forward(self, X: Tensor) -> Tensor:

        #
        return X

    #
    def __repr__(self) -> str:

        #
        return self.__str__()

    #
    def __str__(self) -> str:

        #
        return "MathExpr_ABSTRACT_CLASS"

    #
    def to_latex(self) -> str:

        #
        return ""




###############################################################################
################################ MATH EXPR LIST ###############################
###############################################################################


#
class MathExprEltList(MathExpr):

    #
    def __init__(self, parent: Optional[MathExpr] = None) -> None:

        #
        super().__init__(parent=parent)

        #
        self.elts: list[MathExpr] = []


    #
    def duplicate_lst(self, new_expr: "MathExprEltList") -> list[MathExpr]:

        #
        return [elt.duplicate(parent=new_expr) for elt in self.elts]


    #
    def duplicate(self, parent: Optional[MathExpr] = None) -> "MathExprEltList":

        #
        new_expr: MathExprEltList = MathExprEltList(parent=parent)

        #
        new_expr.elts = self.duplicate_lst(new_expr=new_expr)

        #
        return new_expr


    #
    def actions(self, history: list[MathExpr]) -> list[tuple[str, list[MathExpr]]]:

        #
        hist: list[MathExpr] = history + [self]

        #
        res_actions: list[tuple[str, list[MathExpr]]] = [ ("add_element", hist) ]

        #
        for i, elt in enumerate(self.elts):

            #
            # res_actions.append( (f"remove_element_{i}", hist) )

            #
            res_actions += elt.actions(history=hist)

        #
        return res_actions

    #
    def get_compose_parameters(self) -> list[nn.Parameter]:

        #
        params: list[nn.Parameter] = []

        #
        for elt in self.elts:

            #
            params += elt.parameters()

        #
        return params




###############################################################################
################################## POLYNOMIAL #################################
###############################################################################

#
class MathExprEltList_Sum(MathExprEltList):

    #
    def __init__(self, parent: Optional[MathExpr] = None) -> None:

        #
        super().__init__(parent=parent)


    #
    def duplicate(self, parent: Optional[MathExpr] = None) -> "MathExprEltList_Sum":

        #
        new_expr: MathExprEltList_Sum = MathExprEltList_Sum(parent=parent)

        #
        new_expr.elts = self.duplicate_lst(new_expr=new_expr)

        #
        return new_expr

    #
    def forward(self, X: Tensor) -> Tensor:

        #
        res: Tensor = torch.zeros_like(X)

        #
        for elt in self.elts:

            #
            res += elt(X)

        #
        return res

    #
    def __str__(self) -> str:

        #
        return f"Sum({[elt.__str__() for elt in self.elts]})"

    #
    def to_latex(self) -> str:

        #
        if len(self.elts) == 0:

            #
            return "0"

        #
        return f"({") + (".join([elt.to_latex() for elt in self.elts])})"




###############################################################################
################################## POLYNOMIAL #################################
###############################################################################

#
class MathExprEltList_Prod(MathExprEltList):

    #
    def __init__(self, parent: Optional[MathExpr] = None) -> None:

        #
        super().__init__(parent=parent)


    #
    def duplicate(self, parent: Optional[MathExpr] = None) -> "MathExprEltList_Prod":

        #
        new_expr: MathExprEltList_Prod = MathExprEltList_Prod(parent=parent)

        #
        new_expr.elts = self.duplicate_lst(new_expr=new_expr)

        #
        return new_expr

    #
    def forward(self, X: Tensor) -> Tensor:

        #
        res: Tensor = torch.ones_like(X)

        #
        for elt in self.elts:

            #
            res *= elt(X)

        #
        return res

    #
    def __str__(self) -> str:

        #
        return f"Product({[elt.__str__() for elt in self.elts]})"

    #
    def to_latex(self) -> str:

        #
        if len(self.elts) == 0:

            #
            return "1"

        #
        return f"({") \\times (".join([elt.to_latex() for elt in self.elts])})"



###############################################################################
################################### CONSTANT ##################################
###############################################################################

#
class MathExpr_Const(MathExpr):

    #
    def __init__(self, parent: Optional[MathExpr] = None) -> None:

        #
        super().__init__(parent=parent)

        #
        self.const: nn.Parameter = nn.Parameter(torch.randn((1,)))


    #
    def duplicate(self, parent: Optional[MathExpr] = None) -> "MathExpr_Const":

        #
        return MathExpr_Const(parent=parent)


    #
    def get_self_parameters(self) -> list[nn.Parameter]:

        #
        return [self.const]

    #
    def forward(self, X: Tensor) -> Tensor:

        #
        return self.const.data

    #
    def __str__(self) -> str:

        #
        return bd(self.const)

    #
    def to_latex(self) -> str:

        #
        return self.__str__()



###############################################################################
############################# MATH EXPR FUNCTIONS #############################
###############################################################################


#
class MathExpr_Composed(MathExpr):

    #
    def __init__(self, parent: Optional[MathExpr] = None) -> None:

        #
        super().__init__(parent=parent)

        #
        self.composed: Optional[MathExpr] = None


    #
    def duplicate_composed(self, new_expr: "MathExpr_Composed") -> Optional[MathExpr]:

        #
        return None if self.composed is None else self.composed.duplicate(parent=new_expr)


    #
    def duplicate(self, parent: Optional[MathExpr] = None) -> "MathExpr_Composed":

        #
        new_expr: MathExpr_Composed = MathExpr_Composed(parent=parent)

        #
        new_expr.composed = self.duplicate_composed(new_expr=new_expr)

        #
        return new_expr


    #
    def actions(self, history: list[MathExpr]) -> list[tuple[str, list[MathExpr]]]:

        #
        hist: list[MathExpr] = history + [self]

        #
        if self.composed is not None:

            #
            composed_actions: list[tuple[str, list[MathExpr]]] = self.composed.actions(history=hist)

            #
            return composed_actions + [] # [("remove_composed", hist)]

        else:

            #
            return [("add_compose", hist)]


    #
    def get_compose_parameters(self) -> list[nn.Parameter]:

        #
        if self.composed is not None:

            #
            return self.composed.parameters()

        #
        return []

    #
    def compose_forward(self, X: Tensor) -> Tensor:

        #
        if self.composed is not None:

            #
            return self.composed(X)

        #
        return X

    #
    def compose_str(self) -> str:

        #
        if self.composed is not None:

            #
            return f"({self.composed.__str__()})"

        #
        return "X"

    #
    def forward(self, X: Tensor) -> Tensor:

        #
        return self.compose_forward(X)

    #
    def to_latex_composed(self) -> str:

        #
        if self.composed is not None:

            #
            return f"({self.composed.to_latex()})"

        #
        else:

            #
            return "x"



###############################################################################
################################## POLYNOMIAL #################################
###############################################################################

#
class MathExpr_PolynomialTerm(MathExpr_Composed):

    #
    def __init__(self, parent: Optional[MathExpr] = None) -> None:

        #
        super().__init__(parent=parent)

        #
        self.factor: nn.Parameter = nn.Parameter(torch.randn((1,)))
        self.power: nn.Parameter = nn.Parameter(torch.randn((1,)))


    #
    def duplicate(self, parent: Optional[MathExpr] = None) -> "MathExpr_PolynomialTerm":

        #
        new_expr: MathExpr_PolynomialTerm = MathExpr_PolynomialTerm(parent=parent)

        #
        new_expr.composed = self.duplicate_composed(new_expr=new_expr)

        #
        return new_expr

    #
    def parameters(self) -> list[nn.Parameter]:

        #
        return [self.factor, self.power]

    #
    def forward(self, X: Tensor) -> Tensor:

        #
        X = self.compose_forward(X)

        #
        return self.factor * X ** ( torch.ceil(1.0 + torch.abs(self.power)) )

    #
    def __str__(self) -> str:

        #
        return f"{bd(self.factor)} * {self.compose_str()} ^ {bd(torch.ceil(1.0 + torch.abs(self.power)))}"

    #
    def to_latex(self) -> str:

        #
        return f"{bd(self.factor)} \\cdot ({self.to_latex_composed()}) ^ {{ {bd(torch.ceil(1.0 + torch.abs(self.power)))} }}"



###############################################################################
################################## POLYNOMIAL #################################
###############################################################################

#
class MathExpr_ExponentialTerm(MathExpr_Composed):

    #
    def __init__(self, parent: Optional[MathExpr] = None) -> None:

        #
        super().__init__(parent=parent)

        #
        self.factor: nn.Parameter = nn.Parameter(torch.randn((1,)))
        self.base: nn.Parameter = nn.Parameter(torch.randn((1,)))


    #
    def duplicate(self, parent: Optional[MathExpr] = None) -> "MathExpr_ExponentialTerm":

        #
        new_expr: MathExpr_ExponentialTerm = MathExpr_ExponentialTerm(parent=parent)

        #
        new_expr.composed = self.duplicate_composed(new_expr=new_expr)

        #
        return new_expr

    #
    def parameters(self) -> list[nn.Parameter]:

        #
        return [self.base, self.factor]

    #
    def forward(self, X: Tensor) -> Tensor:

        #
        X = self.compose_forward(X)

        #
        return self.factor * ( (1.0 + torch.abs(self.base) ) ** X)

    #
    def __str__(self) -> str:

        #
        return f"{bd(self.factor)} * {bd(1.0 + torch.abs(self.base))} ^ {self.compose_str()}"

    #
    def to_latex(self) -> str:

        #
        return f"{bd(self.factor)} \\cdot {bd(1.0 + torch.abs(self.base))} ^ {self.to_latex_composed()}"




###############################################################################
################################# LOGARITHMIC #################################
###############################################################################

#
class MathExpr_LogarithmTerm(MathExpr_Composed):

    #
    def __init__(self, parent: Optional[MathExpr] = None) -> None:

        #
        super().__init__(parent=parent)

        #
        self.factor: nn.Parameter = nn.Parameter(torch.randn((1,)))
        self.base: nn.Parameter = nn.Parameter(torch.randn((1,)))


    #
    def duplicate(self, parent: Optional[MathExpr] = None) -> "MathExpr_LogarithmTerm":

        #
        new_expr: MathExpr_LogarithmTerm = MathExpr_LogarithmTerm(parent=parent)

        #
        new_expr.composed = self.duplicate_composed(new_expr=new_expr)

        #
        return new_expr

    #
    def parameters(self) -> list[nn.Parameter]:

        #
        return [self.base, self.factor]

    #
    def forward(self, X: Tensor) -> Tensor:

        #
        X = self.compose_forward(X)

        #
        return self.factor * log( X = 1.0 + torch.abs(X), base = 1.0 + torch.abs(self.base) )

    #
    def __str__(self) -> str:

        #
        return f"{bd(self.factor)} * log_{bd(1.0 + torch.abs(self.base))}({self.compose_str()})"

    #
    def to_latex(self) -> str:

        #
        return f"{bd(self.factor)} \\cdot \\log_{{ {bd(1.0 + torch.abs(self.base))} }}({self.to_latex_composed()})"





###############################################################################
################################# INVERSE #################################
###############################################################################


#
class MathExpr_InverseTerm(MathExpr_Composed):

    #
    def __init__(self, parent: Optional[MathExpr] = None) -> None:

        #
        super().__init__(parent=parent)

        #
        self.factor: nn.Parameter = nn.Parameter(torch.randn((1,)))


    #
    def duplicate(self, parent: Optional[MathExpr] = None) -> "MathExpr_InverseTerm":

        #
        new_expr: MathExpr_InverseTerm = MathExpr_InverseTerm(parent=parent)

        #
        new_expr.composed = self.duplicate_composed(new_expr=new_expr)

        #
        return new_expr

    #
    def parameters(self) -> list[nn.Parameter]:

        #
        return [self.factor]

    #
    def forward(self, X: Tensor) -> Tensor:

        #
        X = self.compose_forward(X)

        #
        return self.factor / X

    #
    def __str__(self) -> str:

        #
        return f"{bd(self.factor)}/{self.compose_str()}"

    #
    def to_latex(self) -> str:

        #
        return "\\frac{ " + bd(self.factor) + " } { " + self.to_latex_composed() + " }"




###############################################################################
################################# INVERSE #################################
###############################################################################


#
class MathExpr_Sinusoidal(MathExpr_Composed):

    #
    def __init__(self, parent: Optional[MathExpr] = None) -> None:

        #
        super().__init__(parent=parent)

        #
        self.amplitude: nn.Parameter = nn.Parameter(torch.randn((1,)))
        self.frequency: nn.Parameter = nn.Parameter(torch.randn((1,)))
        self.phase: nn.Parameter = nn.Parameter(torch.randn((1,)))


    #
    def duplicate(self, parent: Optional[MathExpr] = None) -> "MathExpr_Sinusoidal":

        #
        new_expr: MathExpr_Sinusoidal = MathExpr_Sinusoidal(parent=parent)

        #
        new_expr.composed = self.duplicate_composed(new_expr=new_expr)

        #
        return new_expr

    #
    def parameters(self) -> list[nn.Parameter]:

        #
        return [self.frequency, self.phase, self.amplitude]

    #
    def forward(self, X: Tensor) -> Tensor:

        #
        X = self.compose_forward(X)

        #
        return self.amplitude * torch.sin( self.frequency * X + self.phase )

    #
    def __str__(self) -> str:

        #
        return f"{bd(self.amplitude)} * sin({bd(self.frequency)} * ({self.to_latex_composed()}) + {bd(self.phase)})"

    #
    def to_latex(self) -> str:

        #
        return f"{bd(self.amplitude)} \\cdot \\sin({bd(self.frequency)} \\cdot ({self.to_latex_composed()}) + {bd(self.phase)})"


###############################################################################
#################################### UTILS ####################################
###############################################################################


#
MATH_EXPRS: dict[str, Callable] = {
    "sum": MathExprEltList_Sum,
    "prod": MathExprEltList_Prod,
    "constant": MathExpr_Const,
    "polynomial": MathExpr_PolynomialTerm,
    # "exponential": MathExpr_ExponentialTerm,
    "logarithmic": MathExpr_LogarithmTerm,
    "inverse": MathExpr_InverseTerm,
    "sinusoidal": MathExpr_Sinusoidal
}


#
MATH_EXPRS_VALUES: list[Callable] = list(MATH_EXPRS.values())


