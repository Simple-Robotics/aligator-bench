import dataclasses
import polars as pl
import uuid
import pprint

from collections import defaultdict
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
    cls_config_count = defaultdict(int)
    for i, example_config in enumerate(instance_configs):
        example = cls(**example_config)
        instance_name = example.name() + f"_{i}"
        for runner_cls, settings in solver_configs:
            runner = runner_cls(settings)
            entry = {"name": runner.name()}
            run_id = uuid.uuid4().hex
            config_count = cls_config_count[runner_cls]
            config_name = f"{runner.name()}_{config_count}"
            print(config_name, settings)
            print(instance_name, example_config)
            res = runner.solve(example, tol)
            print(res)
            entry.update(dataclasses.asdict(res))
            entry["run_id"] = run_id
            entry["instance"] = instance_name
            entry["nsteps"] = example.problem.num_steps
            data_.append(entry)
            suppl_data[run_id] = {
                "solver": {
                    "name": runner.name(),
                    "config_name": config_name,
                    **settings,
                },
                "instance": {"name": instance_name, **example_config},
            }
            pprint.pp(suppl_data[run_id])
            print("-------")
            cls_config_count[runner_cls] += 1

    df = pl.DataFrame(data_)
    df = df.rename({"status": "status_old"})
    df = df.with_columns(status=pl.col("status_old").cast(status_dtype))
    df.drop_in_place("status_old")
    _save_run(df, bench_name)
    with open(RESULTS_DIR / f"{bench_name}.pkl", "wb") as f:
        pickle.dump(suppl_data, f)
    return df
