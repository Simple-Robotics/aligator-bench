import dataclasses
import polars as pl

from typing import Type, Tuple
from .solver_runner import status_dtype
from pathlib import Path

SolverConfig = Tuple[Type, dict]
RESULTS_DIR = Path("results/")


def run_benchmark_configs(
    cls: Type, tol, instance_configs: list[dict], solver_configs: list[SolverConfig]
):
    data_ = []
    for i, config in enumerate(instance_configs):
        example = cls(**config)
        for runner_cls, settings in solver_configs:
            runner = runner_cls(settings)
            entry = {"name": runner.name()}
            print(runner.name(), settings)
            print(example.name(), config)
            res = runner.solve(example, tol)
            print(res)
            print("-------")
            entry.update(dataclasses.asdict(res))
            instance_name = example.name() + f"_{i}"
            entry["instance"] = instance_name
            data_.append(entry)

    df = pl.DataFrame(data_)
    df = df.rename({"status": "status_old"})
    df = df.with_columns(status=pl.col("status_old").cast(status_dtype))
    df.drop_in_place("status_old")
    return df


def save_run(df: pl.DataFrame, bench_name):
    df.write_csv(RESULTS_DIR / f"{bench_name}.csv")
