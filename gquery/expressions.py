"""Vectorized Expression Tree for filters, projections, and aggregations."""

from abc import ABC, abstractmethod
from typing import Any, List, Set
from enum import Enum
from gquery.record_batch import ColumnVector, RecordBatch
from gquery.types import DataType, Boolean, Int64, Float64, Utf8


class BinaryOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    EQ = "=="
    NEQ = "!="
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="
    AND = "AND"
    OR = "OR"


class Expr(ABC):
    """Base class for all query plan expressions."""

    @abstractmethod
    def evaluate(self, batch: RecordBatch) -> ColumnVector:
        pass

    @abstractmethod
    def columns_referenced(self) -> Set[str]:
        pass

    def __add__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr(self, BinaryOp.ADD, _to_expr(other))

    def __sub__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr(self, BinaryOp.SUB, _to_expr(other))

    def __mul__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr(self, BinaryOp.MUL, _to_expr(other))

    def __truediv__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr(self, BinaryOp.DIV, _to_expr(other))

    def __eq__(self, other: Any) -> "BinaryExpr":  # type: ignore[override]
        return BinaryExpr(self, BinaryOp.EQ, _to_expr(other))

    def __ne__(self, other: Any) -> "BinaryExpr":  # type: ignore[override]
        return BinaryExpr(self, BinaryOp.NEQ, _to_expr(other))

    def __lt__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr(self, BinaryOp.LT, _to_expr(other))

    def __le__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr(self, BinaryOp.LTE, _to_expr(other))

    def __gt__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr(self, BinaryOp.GT, _to_expr(other))

    def __ge__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr(self, BinaryOp.GTE, _to_expr(other))

    def __and__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr(self, BinaryOp.AND, _to_expr(other))

    def __or__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr(self, BinaryOp.OR, _to_expr(other))


def _to_expr(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    return LiteralExpr(value)


def col(name: str) -> "ColumnExpr":
    return ColumnExpr(name)


def lit(value: Any) -> "LiteralExpr":
    return LiteralExpr(value)


class ColumnExpr(Expr):
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, batch: RecordBatch) -> ColumnVector:
        return batch.column(self.name)

    def columns_referenced(self) -> Set[str]:
        return {self.name}

    def __repr__(self) -> str:
        return f"col('{self.name}')"


class LiteralExpr(Expr):
    def __init__(self, value: Any):
        self.value = value

    def evaluate(self, batch: RecordBatch) -> ColumnVector:
        val = self.value
        n = batch.num_rows
        if isinstance(val, int):
            dtype: DataType = Int64()
        elif isinstance(val, float):
            dtype = Float64()
        elif isinstance(val, str):
            dtype = Utf8()
        elif isinstance(val, bool):
            dtype = Boolean()
        else:
            dtype = Utf8()
        return ColumnVector(dtype, [val] * n, [val is None] * n)

    def columns_referenced(self) -> Set[str]:
        return set()

    def __repr__(self) -> str:
        return f"lit({repr(self.value)})"


class BinaryExpr(Expr):
    def __init__(self, left: Expr, op: BinaryOp, right: Expr):
        self.left = left
        self.op = op
        self.right = right

    def columns_referenced(self) -> Set[str]:
        return self.left.columns_referenced().union(self.right.columns_referenced())

    def evaluate(self, batch: RecordBatch) -> ColumnVector:
        left_vec = self.left.evaluate(batch)
        right_vec = self.right.evaluate(batch)
        n = batch.num_rows

        res_vals: List[Any] = [None] * n
        res_mask: List[bool] = [False] * n

        op = self.op
        for i in range(n):
            if left_vec.is_null(i) or right_vec.is_null(i):
                res_mask[i] = True
                continue
            l = left_vec.values[i]
            r = right_vec.values[i]

            if op == BinaryOp.ADD:
                res_vals[i] = l + r
            elif op == BinaryOp.SUB:
                res_vals[i] = l - r
            elif op == BinaryOp.MUL:
                res_vals[i] = l * r
            elif op == BinaryOp.DIV:
                res_vals[i] = l / r if r != 0 else None
                if res_vals[i] is None:
                    res_mask[i] = True
            elif op == BinaryOp.EQ:
                res_vals[i] = (l == r)
            elif op == BinaryOp.NEQ:
                res_vals[i] = (l != r)
            elif op == BinaryOp.LT:
                res_vals[i] = (l < r)
            elif op == BinaryOp.LTE:
                res_vals[i] = (l <= r)
            elif op == BinaryOp.GT:
                res_vals[i] = (l > r)
            elif op == BinaryOp.GTE:
                res_vals[i] = (l >= r)
            elif op == BinaryOp.AND:
                res_vals[i] = bool(l and r)
            elif op == BinaryOp.OR:
                res_vals[i] = bool(l or r)

        # Infer return type
        if op in {BinaryOp.EQ, BinaryOp.NEQ, BinaryOp.LT, BinaryOp.LTE, BinaryOp.GT, BinaryOp.GTE, BinaryOp.AND, BinaryOp.OR}:
            return_type = Boolean()
        elif op == BinaryOp.DIV:
            return_type = Float64()
        else:
            return_type = left_vec.data_type

        return ColumnVector(return_type, res_vals, res_mask)

    def __repr__(self) -> str:
        return f"({self.left} {self.op.value} {self.right})"
