"""Rule-based Query Optimizer implementing Predicate Pushdown and Column Pruning."""

from typing import Set, List
from gquery.logical_plan import (
    LogicalPlan,
    ScanNode,
    FilterNode,
    ProjectNode,
    AggregateNode,
    JoinNode,
)


class Optimizer:
    """Applies rule-based optimization passes to logical query plans."""

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        plan = self.push_down_predicates(plan)
        plan = self.prune_columns(plan)
        return plan

    def push_down_predicates(self, plan: LogicalPlan) -> LogicalPlan:
        """Pushes Filter operations down as close to Scan nodes as possible."""
        if isinstance(plan, FilterNode):
            child = self.push_down_predicates(plan.input_plan)
            # Push past Project if possible
            if isinstance(child, ProjectNode):
                # Filter before projecting
                return ProjectNode(
                    input_plan=FilterNode(child.input_plan, plan.predicate),
                    exprs=child.exprs
                )
            return FilterNode(child, plan.predicate)

        elif isinstance(plan, ProjectNode):
            return ProjectNode(self.push_down_predicates(plan.input_plan), plan.exprs)

        elif isinstance(plan, AggregateNode):
            return AggregateNode(
                self.push_down_predicates(plan.input_plan),
                plan.group_exprs,
                plan.agg_exprs
            )

        elif isinstance(plan, JoinNode):
            return JoinNode(
                self.push_down_predicates(plan.left),
                self.push_down_predicates(plan.right),
                plan.on_left,
                plan.on_right,
                plan.join_type
            )

        return plan

    def prune_columns(self, plan: LogicalPlan) -> LogicalPlan:
        """Pushes required column projections down into Scan nodes."""
        required = self._collect_required_columns(plan)
        return self._apply_projection_pushdown(plan, required)

    def _collect_required_columns(self, plan: LogicalPlan) -> Set[str]:
        cols: Set[str] = set()
        if isinstance(plan, ScanNode):
            return {f.name for f in plan.schema().fields}
        elif isinstance(plan, FilterNode):
            cols.update(plan.predicate.columns_referenced())
            cols.update(self._collect_required_columns(plan.input_plan))
        elif isinstance(plan, ProjectNode):
            for expr, _ in plan.exprs:
                cols.update(expr.columns_referenced())
            cols.update(self._collect_required_columns(plan.input_plan))
        elif isinstance(plan, AggregateNode):
            cols.update(plan.group_exprs)
            for _, cname, _ in plan.agg_exprs:
                cols.add(cname)
            cols.update(self._collect_required_columns(plan.input_plan))
        elif isinstance(plan, JoinNode):
            cols.add(plan.on_left)
            cols.add(plan.on_right)
            cols.update(self._collect_required_columns(plan.left))
            cols.update(self._collect_required_columns(plan.right))
        return cols

    def _apply_projection_pushdown(self, plan: LogicalPlan, required: Set[str]) -> LogicalPlan:
        if isinstance(plan, ScanNode):
            all_cols = [f.name for f in plan._schema.fields]
            pruned = [c for c in all_cols if c in required]
            return ScanNode(plan.table_name, plan._schema, projection=pruned if pruned else None)
        elif isinstance(plan, FilterNode):
            return FilterNode(self._apply_projection_pushdown(plan.input_plan, required), plan.predicate)
        elif isinstance(plan, ProjectNode):
            return ProjectNode(self._apply_projection_pushdown(plan.input_plan, required), plan.exprs)
        elif isinstance(plan, AggregateNode):
            return AggregateNode(
                self._apply_projection_pushdown(plan.input_plan, required),
                plan.group_exprs,
                plan.agg_exprs
            )
        elif isinstance(plan, JoinNode):
            return JoinNode(
                self._apply_projection_pushdown(plan.left, required),
                self._apply_projection_pushdown(plan.right, required),
                plan.on_left,
                plan.on_right,
                plan.join_type
            )
        return plan
