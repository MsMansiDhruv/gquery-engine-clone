"""Micro-benchmark comparing row-by-row iteration vs GQuery vectorized batch processing."""

import time
import random
from gquery.types import Schema, Field, Int64, Float64, Utf8
from gquery.record_batch import ColumnVector, RecordBatch
from gquery.expressions import col, lit
from gquery.engine import ExecutionContext


def run_benchmark(n_rows: int = 100_000):
    print(f"=== GQuery Micro-benchmark ({n_rows:,} rows) ===")

    # 1. Generate synthetic data
    random.seed(42)
    order_ids = list(range(1, n_rows + 1))
    customer_ids = [random.randint(1, 1000) for _ in range(n_rows)]
    amounts = [round(random.uniform(10.0, 1000.0), 2) for _ in range(n_rows)]
    statuses = [random.choice(["DELIVERED", "PENDING", "CANCELLED", "SHIPPED"]) for _ in range(n_rows)]

    # 2. Row-oriented baseline (traditional python dict list)
    row_data = [
        {"order_id": order_ids[i], "customer_id": customer_ids[i], "amount": amounts[i], "status": statuses[i]}
        for i in range(n_rows)
    ]

    # Benchmark: Filter (amount > 500 AND status == 'DELIVERED') + sum(amount)
    t0 = time.perf_counter()
    row_sum = 0.0
    row_count = 0
    for r in row_data:
        if r["amount"] > 500.0 and r["status"] == "DELIVERED":
            row_sum += r["amount"]
            row_count += 1
    t_row = time.perf_counter() - t0
    print(f"Row-based execution:       {t_row * 1000:7.2f} ms | matches={row_count:,}, sum={row_sum:,.2f}")

    # 3. Vectorized Columnar GQuery execution
    schema = Schema([
        Field("order_id", Int64()),
        Field("customer_id", Int64()),
        Field("amount", Float64()),
        Field("status", Utf8()),
    ])
    batch = RecordBatch(
        schema,
        [
            ColumnVector(Int64(), order_ids),
            ColumnVector(Int64(), customer_ids),
            ColumnVector(Float64(), amounts),
            ColumnVector(Utf8(), statuses),
        ]
    )

    ctx = ExecutionContext()
    ctx.register_batch("orders", batch)

    t0 = time.perf_counter()
    df = (
        ctx.from_table("orders")
        .filter((col("amount") > 500.0) & (col("status") == "DELIVERED"))
        .select("amount")
    )
    result = df.collect()
    vec_sum = sum(result.column("amount").values)
    vec_count = result.num_rows
    t_vec = time.perf_counter() - t0

    print(f"GQuery Vectorized execution: {t_vec * 1000:7.2f} ms | matches={vec_count:,}, sum={vec_sum:,.2f}")
    speedup = t_row / t_vec if t_vec > 0 else 1.0
    print(f"Execution Speedup:           {speedup:.2f}x")


if __name__ == "__main__":
    run_benchmark(50_000)
