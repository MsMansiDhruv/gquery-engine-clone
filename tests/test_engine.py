"""Unit tests for GQuery vectorized query engine."""

import unittest
from gquery.types import Schema, Field, Int64, Float64, Utf8, Boolean
from gquery.record_batch import ColumnVector, RecordBatch
from gquery.expressions import col, lit, BinaryOp
from gquery.engine import ExecutionContext


class TestGQueryEngine(unittest.TestCase):

    def setUp(self):
        # Create sample orders batch
        schema = Schema([
            Field("order_id", Int64()),
            Field("customer_id", Int64()),
            Field("amount", Float64()),
            Field("status", Utf8()),
        ])
        orders_batch = RecordBatch(
            schema,
            [
                ColumnVector(Int64(), [101, 102, 103, 104, 105]),
                ColumnVector(Int64(), [1, 2, 1, 3, 2]),
                ColumnVector(Float64(), [250.0, 120.0, 450.0, 80.0, 310.0]),
                ColumnVector(Utf8(), ["DELIVERED", "PENDING", "DELIVERED", "CANCELLED", "DELIVERED"]),
            ]
        )

        # Create sample customers batch
        cust_schema = Schema([
            Field("cid", Int64()),
            Field("customer_name", Utf8()),
            Field("region", Utf8()),
        ])
        cust_batch = RecordBatch(
            cust_schema,
            [
                ColumnVector(Int64(), [1, 2, 3]),
                ColumnVector(Utf8(), ["Alice", "Bob", "Charlie"]),
                ColumnVector(Utf8(), ["North", "South", "West"]),
            ]
        )

        self.ctx = ExecutionContext()
        self.ctx.register_batch("orders", orders_batch)
        self.ctx.register_batch("customers", cust_batch)

    def test_filter_and_projection(self):
        df = (
            self.ctx.from_table("orders")
            .filter((col("amount") > 200.0) & (col("status") == "DELIVERED"))
            .select("order_id", "amount")
        )
        result = df.collect()
        self.assertEqual(result.num_rows, 3)
        self.assertEqual(result.column("order_id").values, [101, 103, 105])
        self.assertEqual(result.column("amount").values, [250.0, 450.0, 310.0])

    def test_hash_aggregation(self):
        df = (
            self.ctx.from_table("orders")
            .filter(col("status") == "DELIVERED")
            .aggregate(
                group_by=["customer_id"],
                aggs=[
                    ("count", "order_id", "total_orders"),
                    ("sum", "amount", "total_spend"),
                ]
            )
        )
        result = df.collect()
        res_dict = result.to_pydict()
        # customer 1 has 2 delivered orders (250 + 450 = 700)
        # customer 2 has 1 delivered order (310)
        self.assertEqual(result.num_rows, 2)
        idx_1 = res_dict["customer_id"].index(1)
        self.assertEqual(res_dict["total_orders"][idx_1], 2)
        self.assertEqual(res_dict["total_spend"][idx_1], 700.0)

    def test_hash_join(self):
        orders = self.ctx.from_table("orders")
        customers = self.ctx.from_table("customers")

        df = (
            orders.join(customers, on_left="customer_id", on_right="cid")
            .filter(col("status") == "DELIVERED")
            .select("order_id", "customer_name", "amount")
        )
        result = df.collect()
        self.assertEqual(result.num_rows, 3)
        names = result.column("customer_name").values
        self.assertIn("Alice", names)
        self.assertIn("Bob", names)

    def test_optimizer_predicate_pushdown(self):
        df = (
            self.ctx.from_table("orders")
            .select("order_id", "amount")
            .filter(col("amount") > 100.0)
        )
        explain_str = df.explain(optimize=True)
        self.assertIn("ScanNode", explain_str)
        self.assertIn("FilterNode", explain_str)
        self.assertIn("ProjectNode", explain_str)


    def test_arithmetic_projection(self):
        df = (
            self.ctx.from_table("orders")
            .filter(col("order_id") == 101)
            .select("order_id", (col("amount") * lit(1.10), "amount_with_tax"))
        )
        result = df.collect()
        self.assertEqual(result.num_rows, 1)
        self.assertAlmostEqual(result.column("amount_with_tax").values[0], 275.0, places=2)

    def test_null_mask_propagation(self):
        schema = Schema([Field("id", Int64()), Field("val", Float64(), nullable=True)])
        batch = RecordBatch(
            schema,
            [
                ColumnVector(Int64(), [1, 2, 3]),
                ColumnVector(Float64(), [10.0, 20.0, 30.0], null_mask=[False, True, False]),
            ]
        )
        ctx = ExecutionContext()
        ctx.register_batch("nullable_data", batch)

        df = ctx.from_table("nullable_data").filter(col("val") > 15.0)
        res = df.collect()
        # Row 2 is null so it evaluates to null and should not pass predicate > 15.0
        self.assertEqual(res.num_rows, 1)
        self.assertEqual(res.column("id").values, [3])

    def test_empty_result_batch(self):
        df = self.ctx.from_table("orders").filter(col("amount") > 99999.0)
        res = df.collect()
        self.assertEqual(res.num_rows, 0)
        self.assertEqual(len(res.schema), 4)


if __name__ == "__main__":
    unittest.main()

