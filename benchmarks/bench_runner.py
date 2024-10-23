import dataclasses

from typing import Type, Tuple

SolverConfig = Tuple[Type, dict]


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

    return data_
