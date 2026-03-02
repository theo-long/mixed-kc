from typing import Literal

InequalityLiteral = Literal["<", "<=", ">", ">="]
inequality_flip_mapping: dict[InequalityLiteral, InequalityLiteral] = {
    ">": "<",
    "<": ">",
    "<=": ">=",
    ">=": "<=",
}
