"""Physical Operators implementing vectorized batch execution."""

from abc import ABC, abstractmethod
from typing import Iterator, List, Dict, Any, Tuple
from collections import defaultdict
from gquery.types import Schema, Field, Int64, Float64, Utf8, Boolean
from gquery.record_batch import RecordBatch, ColumnVector
from gquery.expressions import Expr


class PhysicalPlan(ABC):
    @abstractmethod
    def schema(self) -> Schema:
        pass

    @abstractmethod
    def execute(self) -> Iterator[RecordBatch]:
        """Stream record batches using the vectorized iterator model."""
        pass


class ScanExec(PhysicalPlan):
    def __init__(self, batches: List[RecordBatch], projection: List[str] = None):
        self.batches = batches
        self.projection = projection
        if batches:
            if projection:
                self._schema = batches[0].schema.project(projection)
            else:
                self._schema = batches[0].schema
        else:
            self._schema = Schema([])

    def schema(self) -> Schema:
        return self._schema

    def execute(self) -> Iterator[RecordBatch]:
        for batch in self.batches:
            if self.projection:
                yield batch.project(self.projection)
            else:
                yield batch


class FilterExec(PhysicalPlan):
    def __init__(self, input_plan: PhysicalPlan, predicate: Expr):
        self.input_plan = input_plan
        self.predicate = predicate

    def schema(self) -> Schema:
        return self.input_plan.schema()

    def execute(self) -> Iterator[RecordBatch]:
        for batch in self.input_plan.execute():
            # Vectorized predicate evaluation produces a boolean ColumnVector
            bool_vector = self.predicate.evaluate(batch)
            selection_mask = [
                bool(bool_vector.values[i]) and not bool_vector.is_null(i)
                for i in range(batch.num_rows)
            ]
            yield batch.filter(selection_mask)


class ProjectExec(PhysicalPlan):
    def __init__(self, input_plan: PhysicalPlan, exprs: List[Tuple[Expr, str]], output_schema: Schema):
        self.input_plan = input_plan
        self.exprs = exprs
        self._schema = output_schema

    def schema(self) -> Schema:
        return self._schema

    def execute(self) -> Iterator[RecordBatch]:
        for batch in self.input_plan.execute():
            eval_cols = []
            for expr, _ in self.exprs:
                vec = expr.evaluate(batch)
                eval_cols.append(vec)
            yield RecordBatch(self._schema, eval_cols)


class HashAggregateExec(PhysicalPlan):
    """Vectorized Hash Aggregate Operator."""

    def __init__(
        self,
        input_plan: PhysicalPlan,
        group_keys: List[str],
        aggs: List[Tuple[str, str, str]],  # (func, column, alias)
        output_schema: Schema,
    ):
        self.input_plan = input_plan
        self.group_keys = group_keys
        self.aggs = aggs
        self._schema = output_schema

    def schema(self) -> Schema:
        return self._schema

    def execute(self) -> Iterator[RecordBatch]:
        # Hash table accumulator: key_tuple -> list of accumulator states
        hash_table: Dict[Tuple, List[Any]] = defaultdict(lambda: [
            {"count": 0, "sum": 0.0, "min": float("inf"), "max": float("-inf")}
            for _ in self.aggs
        ])

        for batch in self.input_plan.execute():
            n = batch.num_rows
            group_vectors = [batch.column(k) for k in self.group_keys]
            agg_vectors = [batch.column(col_name) for _, col_name, _ in self.aggs]

            for i in range(n):
                key = tuple(gv.get(i) for gv in group_vectors)
                accs = hash_table[key]

                for agg_idx, (func, _, _) in enumerate(self.aggs):
                    val = agg_vectors[agg_idx].get(i)
                    if val is None:
                        continue
                    acc = accs[agg_idx]
                    acc["count"] += 1
                    acc["sum"] += float(val)
                    if val < acc["min"]:
                        acc["min"] = val
                    if val > acc["max"]:
                        acc["max"] = val

        # Materialize aggregated batches
        result_rows = list(hash_table.items())
        if not result_rows:
            return

        # Build column vectors
        group_cols_data: Dict[str, List[Any]] = {k: [] for k in self.group_keys}
        agg_cols_data: Dict[str, List[Any]] = {alias: [] for _, _, alias in self.aggs}

        for key_tuple, accs in result_rows:
            for k_idx, k_name in enumerate(self.group_keys):
                group_cols_data[k_name].append(key_tuple[k_idx])

            for a_idx, (func, _, alias) in enumerate(self.aggs):
                f = func.lower()
                acc = accs[a_idx]
                if f == "count":
                    agg_cols_data[alias].append(acc["count"])
                elif f == "sum":
                    agg_cols_data[alias].append(acc["sum"])
                elif f == "avg":
                    agg_cols_data[alias].append(acc["sum"] / acc["count"] if acc["count"] > 0 else None)
                elif f == "min":
                    agg_cols_data[alias].append(acc["min"] if acc["count"] > 0 else None)
                elif f == "max":
                    agg_cols_data[alias].append(acc["max"] if acc["count"] > 0 else None)

        output_columns = []
        for field in self._schema.fields:
            if field.name in group_cols_data:
                vals = group_cols_data[field.name]
            else:
                vals = agg_cols_data[field.name]
            output_columns.append(ColumnVector(field.data_type, vals, [v is None for v in vals]))

        yield RecordBatch(self._schema, output_columns)


class HashJoinExec(PhysicalPlan):
    """Vectorized In-Memory Hash Join Operator."""

    def __init__(
        self,
        left: PhysicalPlan,
        right: PhysicalPlan,
        on_left: str,
        on_right: str,
        output_schema: Schema,
    ):
        self.left = left
        self.right = right
        self.on_left = on_left
        self.on_right = on_right
        self._schema = output_schema

    def schema(self) -> Schema:
        return self._schema

    def execute(self) -> Iterator[RecordBatch]:
        # Build Phase on right table
        hash_table: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for r_batch in self.right.execute():
            key_col = r_batch.column(self.on_right)
            r_dict = r_batch.to_pydict()
            for i in range(r_batch.num_rows):
                if not key_col.is_null(i):
                    k = key_col.values[i]
                    row_data = {col_name: r_dict[col_name][i] for col_name in r_dict if col_name != self.on_right}
                    hash_table[k].append(row_data)

        # Probe Phase on left stream
        for l_batch in self.left.execute():
            key_col = l_batch.column(self.on_left)
            l_dict = l_batch.to_pydict()

            joined_data: Dict[str, List[Any]] = {f.name: [] for f in self._schema.fields}

            for i in range(l_batch.num_rows):
                if key_col.is_null(i):
                    continue
                k = key_col.values[i]
                matching_rights = hash_table.get(k, [])

                for r_row in matching_rights:
                    for l_col_name in l_dict:
                        joined_data[l_col_name].append(l_dict[l_col_name][i])
                    for r_col_name, r_val in r_row.items():
                        joined_data[r_col_name].append(r_val)

            num_joined_rows = len(joined_data[self._schema.fields[0].name])
            if num_joined_rows > 0:
                cols = []
                for field in self._schema.fields:
                    vals = joined_data[field.name]
                    cols.append(ColumnVector(field.data_type, vals, [v is None for v in vals]))
                yield RecordBatch(self._schema, cols)
