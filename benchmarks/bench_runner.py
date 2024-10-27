import dataclasses
import polars as pl
import uuid

from collections import defaultdict
from typing import Type, Tuple
from .solver_runner import status_dtype
from pathlib import Path

SolverConfig = Tuple[Type, dict]
RESULTS_DIR = Path("results/")


def run_single_instance(
    bench_name, cls, tol, instance_name, example_config, solver_configs
):
    import pickle

    example = cls(**example_config)
    data_ = []
    suppl_data = {}
    cls_config_count = defaultdict(int)
    for runner_cls, settings in solver_configs:
        runner = runner_cls(settings)
        run_id = uuid.uuid4().hex
        entry = {"name": runner.name(), "run_id": run_id, "instance": instance_name}
        config_count = cls_config_count[runner_cls]
        config_name = f"{runner.name()}:{config_count}"
        print("Solver config:", config_name, settings)
        print("Problem instance:", instance_name, example_config)
        res = runner.solve(example, tol)
        print(res)
        entry.update(dataclasses.asdict(res))
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
        print("-------")
        cls_config_count[runner_cls] += 1

    subpath = RESULTS_DIR / bench_name
    subpath.mkdir(parents=True, exist_ok=True)

    df = pl.DataFrame(data_)
    df = df.with_columns(status=pl.col("status").cast(status_dtype))

    df.write_csv(subpath / f"{instance_name}.csv")
    with open(subpath / f"{instance_name}.pkl", "wb") as f:
        pickle.dump(suppl_data, f)

    with pl.Config(tbl_rows=-1, tbl_cols=12):
        print(df)


def run_benchmark_configs(
    bench_name,
    cls: Type,
    tol,
    instance_configs: list[dict],
    solver_configs: list[SolverConfig],
):
    from multiprocessing import Pool

    pool = Pool(3)
    inputs = []
    for i, example_config in enumerate(instance_configs):
        instance_name = cls.name() + f"_{i}"
        # run_single_instance(
        #     bench_name, cls, tol, instance_name, example_config, solver_configs
        # )
        inputs.append(
            (bench_name, cls, tol, instance_name, example_config, solver_configs)
        )

    pool.starmap(run_single_instance, inputs)
    pool.close()
    pool.join()
