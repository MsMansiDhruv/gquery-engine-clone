"""Top-level Query Execution Context and Fluent Query Builder."""

from typing import List, Dict, Tuple, Optional, Any
from gquery.types import Schema
from gquery.record_batch import RecordBatch, ColumnVector
from gquery.expressions import Expr, col
from gquery.logical_plan import (
    LogicalPlan,
    ScanNode,
    FilterNode,
    ProjectNode,
    AggregateNode,
    JoinNode,
)
from gquery.physical_plan import (
    PhysicalPlan,
    ScanExec,
    FilterExec,
    ProjectExec,
    HashAggregateExec,
    HashJoinExec,
)
from gquery.optimizer import Optimizer


class DataFrame:
    """Fluent API representing a deferred computation on an ExecutionContext."""

    def __init__(self, ctx: "ExecutionContext", plan: LogicalPlan):
        self.ctx = ctx
        self.plan = plan

    def filter(self, predicate: Expr) -> "DataFrame":
        return DataFrame(self.ctx, FilterNode(self.plan, predicate))

    def select(self, *exprs: Any) -> "DataFrame":
        """Accepts column name strings or (Expr, alias) tuples."""
        named_exprs: List[Tuple[Expr, str]] = []
        for e in exprs:
            if isinstance(e, str):
                named_exprs.append((col(e), e))
            elif isinstance(e, tuple) and len(e) == 2:
                named_exprs.append(e)
            elif isinstance(e, Expr):
                named_exprs.append((e, repr(e)))
            else:
                raise ValueError(f"Unsupported select item: {e}")
        return DataFrame(self.ctx, ProjectNode(self.plan, named_exprs))

    def join(self, other: "DataFrame", on_left: str, on_right: str, join_type: str = "inner") -> "DataFrame":
        return DataFrame(self.ctx, JoinNode(self.plan, other.plan, on_left, on_right, join_type))

    def aggregate(self, group_by: List[str], aggs: List[Tuple[str, str, str]]) -> "DataFrame":
        """aggs: [("sum", "amount", "total_amount"), ("count", "order_id", "order_count")]"""
        return DataFrame(self.ctx, AggregateNode(self.plan, group_by, aggs))

    def explain(self, optimize: bool = True) -> str:
        lines = ["=== Logical Plan ===", self.plan.display(0)]
        if optimize:
            opt_plan = self.ctx.optimizer.optimize(self.plan)
            lines.extend(["", "=== Optimized Logical Plan ===", opt_plan.display(0)])
            phys_plan = self.ctx.to_physical(opt_plan)
            lines.extend(["", "=== Physical Execution Tree ===", f"PhysicalPlan({phys_plan.__class__.__name__})"])
        return "\n".join(lines)

    def collect(self) -> RecordBatch:
        return self.ctx.execute_plan(self.plan)

    def show(self, limit: int = 10) -> None:
        batch = self.collect()
        print(batch.show(limit=limit))


class ExecutionContext:
    """Manages catalog schemas, registered table batches, and execution planning."""

    def __init__(self):
        self.tables: Dict[str, List[RecordBatch]] = {}
        self.optimizer = Optimizer()

    def register_batches(self, table_name: str, batches: List[RecordBatch]) -> None:
        if not batches:
            raise ValueError("Must register at least one batch")
        self.tables[table_name] = batches

    def register_batch(self, table_name: str, batch: RecordBatch) -> None:
        self.register_batches(table_name, [batch])

    def from_table(self, table_name: str) -> DataFrame:
        if table_name not in self.tables:
            raise KeyError(f"Table '{table_name}' is not registered in context")
        batches = self.tables[table_name]
        schema = batches[0].schema
        return DataFrame(self, ScanNode(table_name, schema))

    def to_physical(self, plan: LogicalPlan) -> PhysicalPlan:
        """Translate logical plan node to physical operator."""
        if isinstance(plan, ScanNode):
            batches = self.tables[plan.table_name]
            return ScanExec(batches, projection=plan.projection)
        elif isinstance(plan, FilterNode):
            return FilterExec(self.to_physical(plan.input_plan), plan.predicate)
        elif isinstance(plan, ProjectNode):
            return ProjectExec(self.to_physical(plan.input_plan), plan.exprs, plan.schema())
        elif isinstance(plan, AggregateNode):
            return HashAggregateExec(
                self.to_physical(plan.input_plan),
                plan.group_exprs,
                plan.agg_exprs,
                plan.schema()
            )
        elif isinstance(plan, JoinNode):
            return HashJoinExec(
                self.to_physical(plan.left),
                self.to_physical(plan.right),
                plan.on_left,
                plan.on_right,
                plan.schema()
            )
        raise NotImplementedError(f"No physical mapping for {plan.__class__.__name__}")

    def execute_plan(self, logical_plan: LogicalPlan, optimize: bool = True) -> RecordBatch:
        if optimize:
            logical_plan = self.optimizer.optimize(logical_plan)

        physical_plan = self.to_physical(logical_plan)
        batches = list(physical_plan.execute())

        if not batches:
            # Return empty batch with plan schema
            return RecordBatch(
                physical_plan.schema(),
                [ColumnVector(f.data_type, []) for f in physical_plan.schema().fields]
            )

        if len(batches) == 1:
            return batches[0]

        # Concatenate multiple batches into a single RecordBatch
        schema = batches[0].schema
        merged_cols: List[ColumnVector] = []
        for f_idx, field in enumerate(schema.fields):
            all_vals: List[Any] = []
            all_masks: List[bool] = []
            for b in batches:
                col_vec = b.columns[f_idx]
                all_vals.extend(col_vec.values)
                all_masks.extend(col_vec.null_mask)
            merged_cols.append(ColumnVector(field.data_type, all_vals, all_masks))

        return RecordBatch(schema, merged_cols)
