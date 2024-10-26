import dataclasses
import polars as pl
import uuid

from typing import Type, Tuple
from .solver_runner import status_dtype
from pathlib import Path

SolverConfig = Tuple[Type, dict]
RESULTS_DIR = Path("results/")


def _save_run(df: pl.DataFrame, bench_name):
    df.write_csv(RESULTS_DIR / f"{bench_name}.csv")


def run_benchmark_configs(
    bench_name,
    cls: Type,
    tol,
    instance_configs: list[dict],
    solver_configs: list[SolverConfig],
):
    import pickle

    data_ = []
    suppl_data = {}
    for i, config in enumerate(instance_configs):
        example = cls(**config)
        instance_name = example.name() + f"_{i}"
        for runner_cls, settings in solver_configs:
            runner = runner_cls(settings)
            entry = {"name": runner.name()}
            print(runner.name(), settings)
            print(example.name(), config)
            res = runner.solve(example, tol)
            print(res)
            print("-------")
            entry.update(dataclasses.asdict(res))
            run_id = uuid.uuid4()
            entry["run_id"] = run_id.hex
            entry["instance"] = instance_name
            entry["nsteps"] = example.problem.num_steps
            data_.append(entry)
            suppl_data[run_id.hex] = {
                "solver": settings.copy(),
                "instance": {"name": instance_name, **config},
            }

    df = pl.DataFrame(data_)
    df = df.rename({"status": "status_old"})
    df = df.with_columns(status=pl.col("status_old").cast(status_dtype))
    df.drop_in_place("status_old")
    _save_run(df, bench_name)
    with open(RESULTS_DIR / f"{bench_name}.pkl", "wb") as f:
        pickle.dump(suppl_data, f)
    return df
