"""Vectorized Columnar RecordBatch and ColumnVector primitives."""

from typing import List, Any, Optional
from gquery.types import Schema, DataType, Field


class ColumnVector:
    """A contiguous chunk of columnar data with optional null masking."""

    def __init__(self, data_type: DataType, values: List[Any], null_mask: Optional[List[bool]] = None):
        self.data_type = data_type
        self.values = list(values)
        self.length = len(values)
        if null_mask is None:
            self.null_mask = [False] * self.length
        else:
            if len(null_mask) != self.length:
                raise ValueError("Length of null_mask must match length of values")
            self.null_mask = list(null_mask)

    def is_null(self, index: int) -> bool:
        return self.null_mask[index]

    def get(self, index: int) -> Any:
        return None if self.null_mask[index] else self.values[index]

    def filter(self, selection_mask: List[bool]) -> "ColumnVector":
        """Vectorized filter applying a boolean bitmask."""
        new_values = [v for v, m in zip(self.values, selection_mask) if m]
        new_mask = [n for n, m in zip(self.null_mask, selection_mask) if m]
        return ColumnVector(self.data_type, new_values, new_mask)

    def slice(self, offset: int, length: int) -> "ColumnVector":
        end = min(offset + length, self.length)
        return ColumnVector(self.data_type, self.values[offset:end], self.null_mask[offset:end])

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        preview = [None if self.null_mask[i] else self.values[i] for i in range(min(5, self.length))]
        ellipsis_str = ", ..." if self.length > 5 else ""
        return f"ColumnVector({self.data_type}, [{', '.join(repr(x) for x in preview)}{ellipsis_str}])"


class RecordBatch:
    """A collection of equal-length ColumnVectors adhering to a defined Schema."""

    def __init__(self, schema: Schema, columns: List[ColumnVector]):
        if len(columns) != len(schema):
            raise ValueError(f"Schema defines {len(schema)} fields, but {len(columns)} columns were passed")
        row_count = len(columns[0]) if columns else 0
        for col in columns:
            if len(col) != row_count:
                raise ValueError(f"Column vector lengths do not match: expected {row_count}, got {len(col)}")

        self.schema = schema
        self.columns = list(columns)
        self.num_rows = row_count

    @property
    def num_columns(self) -> int:
        return len(self.columns)

    def column(self, index_or_name: Any) -> ColumnVector:
        if isinstance(index_or_name, str):
            idx = self.schema.index_of(index_or_name)
            return self.columns[idx]
        return self.columns[index_or_name]

    def filter(self, selection_mask: List[bool]) -> "RecordBatch":
        """Filter all columnar vectors simultaneously using a selection vector."""
        filtered_cols = [col.filter(selection_mask) for col in self.columns]
        return RecordBatch(self.schema, filtered_cols)

    def project(self, column_names: List[str]) -> "RecordBatch":
        """Prune columns to match the projected subset."""
        new_schema = self.schema.project(column_names)
        new_cols = [self.column(name) for name in column_names]
        return RecordBatch(new_schema, new_cols)

    def to_pydict(self) -> dict:
        result = {}
        for idx, field in enumerate(self.schema.fields):
            col = self.columns[idx]
            result[field.name] = [None if col.is_null(i) else col.values[i] for i in range(len(col))]
        return result

    def show(self, limit: int = 10) -> str:
        """Render ASCII table format."""
        headers = [f.name for f in self.schema.fields]
        rows = []
        display_count = min(self.num_rows, limit)
        for i in range(display_count):
            rows.append([str(None if col.is_null(i) else col.values[i]) for col in self.columns])

        col_widths = [max(len(h), max((len(r[idx]) for r in rows), default=0)) for idx, h in enumerate(headers)]
        separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        header_row = "|" + "|".join(f" {h.center(w)} " for h, w in zip(headers, col_widths)) + "|"

        lines = [separator, header_row, separator]
        for row in rows:
            lines.append("|" + "|".join(f" {val.ljust(w)} " for val, w in zip(row, col_widths)) + "|")
        lines.append(separator)
        if self.num_rows > display_count:
            lines.append(f"... ({self.num_rows - display_count} more rows)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"RecordBatch(rows={self.num_rows}, cols={self.num_columns})\n{self.show(limit=5)}"
