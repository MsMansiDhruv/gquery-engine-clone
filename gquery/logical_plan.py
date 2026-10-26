"""Logical Query Plan representations."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from gquery.types import Schema
from gquery.expressions import Expr


class LogicalPlan(ABC):
    @abstractmethod
    def schema(self) -> Schema:
        pass

    @abstractmethod
    def children(self) -> List["LogicalPlan"]:
        pass

    @abstractmethod
    def display(self, indent: int = 0) -> str:
        pass

    def __repr__(self) -> str:
        return self.display(0)


class ScanNode(LogicalPlan):
    def __init__(self, table_name: str, table_schema: Schema, projection: Optional[List[str]] = None):
        self.table_name = table_name
        self._schema = table_schema
        self.projection = projection

    def schema(self) -> Schema:
        if self.projection:
            return self._schema.project(self.projection)
        return self._schema

    def children(self) -> List[LogicalPlan]:
        return []

    def display(self, indent: int = 0) -> str:
        proj_str = f", projection={self.projection}" if self.projection else ""
        return f"{'  ' * indent}ScanNode(table='{self.table_name}'{proj_str})"


class FilterNode(LogicalPlan):
    def __init__(self, input_plan: LogicalPlan, predicate: Expr):
        self.input_plan = input_plan
        self.predicate = predicate

    def schema(self) -> Schema:
        return self.input_plan.schema()

    def children(self) -> List[LogicalPlan]:
        return [self.input_plan]

    def display(self, indent: int = 0) -> str:
        return f"{'  ' * indent}FilterNode(predicate={self.predicate})\n{self.input_plan.display(indent + 1)}"


class ProjectNode(LogicalPlan):
    def __init__(self, input_plan: LogicalPlan, exprs: List[Tuple[Expr, str]]):
        self.input_plan = input_plan
        self.exprs = exprs
        # Build projected schema
        from gquery.types import Field, DataType
        input_schema = input_plan.schema()
        fields = []
        for expr, alias in exprs:
            # Simple type inference fallback
            orig_field = input_schema.field_by_name(alias)
            dtype: DataType = orig_field.data_type if orig_field else input_schema.fields[0].data_type
            fields.append(Field(name=alias, data_type=dtype))
        self._schema = Schema(fields)

    def schema(self) -> Schema:
        return self._schema

    def children(self) -> List[LogicalPlan]:
        return [self.input_plan]

    def display(self, indent: int = 0) -> str:
        expr_str = ", ".join(f"{expr} AS {alias}" for expr, alias in self.exprs)
        return f"{'  ' * indent}ProjectNode([{expr_str}])\n{self.input_plan.display(indent + 1)}"


class AggregateNode(LogicalPlan):
    def __init__(self, input_plan: LogicalPlan, group_exprs: List[str], agg_exprs: List[Tuple[str, str, str]]):
        """
        agg_exprs: list of (func, col_name, output_alias), e.g. ("sum", "amount", "total_amount")
        """
        self.input_plan = input_plan
        self.group_exprs = group_exprs
        self.agg_exprs = agg_exprs

        from gquery.types import Field, Int64, Float64
        input_schema = input_plan.schema()
        fields = []
        for g in group_exprs:
            fields.append(input_schema.field_by_name(g))  # type: ignore[arg-type]
        for func, cname, alias in agg_exprs:
            dtype = Int64() if func.lower() == "count" else Float64()
            fields.append(Field(name=alias, data_type=dtype))
        self._schema = Schema(fields)

    def schema(self) -> Schema:
        return self._schema

    def children(self) -> List[LogicalPlan]:
        return [self.input_plan]

    def display(self, indent: int = 0) -> str:
        aggs = ", ".join(f"{f}({c}) AS {a}" for f, c, a in self.agg_exprs)
        return f"{'  ' * indent}AggregateNode(group_by={self.group_exprs}, aggs=[{aggs}])\n{self.input_plan.display(indent + 1)}"


class JoinNode(LogicalPlan):
    def __init__(self, left: LogicalPlan, right: LogicalPlan, on_left: str, on_right: str, join_type: str = "inner"):
        self.left = left
        self.right = right
        self.on_left = on_left
        self.on_right = on_right
        self.join_type = join_type

        # Combined schema
        fields = list(left.schema().fields)
        for f in right.schema().fields:
            if f.name != on_right:
                fields.append(f)
        self._schema = Schema(fields)

    def schema(self) -> Schema:
        return self._schema

    def children(self) -> List[LogicalPlan]:
        return [self.left, self.right]

    def display(self, indent: int = 0) -> str:
        return (
            f"{'  ' * indent}JoinNode({self.join_type.upper()} ON {self.on_left}={self.on_right})\n"
            f"{self.left.display(indent + 1)}\n"
            f"{self.right.display(indent + 1)}"
        )
