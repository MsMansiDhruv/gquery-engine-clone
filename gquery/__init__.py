"""
GQuery: Vectorized Columnar Query Engine Laboratory
Exploring database internals, columnar record batches, logical plan optimization, and vectorized physical operators.
"""

from gquery.types import DataType, Int64, Float64, Utf8, Boolean, Field, Schema
from gquery.record_batch import ColumnVector, RecordBatch
from gquery.expressions import col, lit, Expr, BinaryOp
from gquery.engine import ExecutionContext

__all__ = [
    "DataType",
    "Int64",
    "Float64",
    "Utf8",
    "Boolean",
    "Field",
    "Schema",
    "ColumnVector",
    "RecordBatch",
    "col",
    "lit",
    "Expr",
    "BinaryOp",
    "ExecutionContext",
]
