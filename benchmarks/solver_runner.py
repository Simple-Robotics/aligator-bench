import aligator
import numpy as np
import polars

from aligator import SolverProxDDP, TrajOptProblem
from enum import Enum, auto
from dataclasses import dataclass


class Status(Enum):
    CONVERGED = auto()
    MAXITERATIONS = auto()
    ERROR = auto()


status_dtype = polars.Enum([e.name for e in Status])


@dataclass
class Result:
    status: str
    traj_cost: float
    niter: int
    prim_feas: float
    dual_feas: float


def default_initialize_rollout(prob: TrajOptProblem):
    us_init = [np.zeros(s.nu) for s in prob.stages]
    xs_init = aligator.rollout([s.dynamics for s in prob.stages], prob.x0_init, us_init)
    return xs_init, us_init


class ProxDdpRunner:
    def __init__(self, settings={}):
        self._settings = settings

    def solve(self, example, tol: float) -> Result:
        prob: TrajOptProblem = example.problem
        solver = SolverProxDDP(tol)
        solver.mu_init = self._settings["mu_init"]  # required param
        solver.linesearch.avg_eta = 0.5
        xs_init = []
        us_init = []
        for param, value in self._settings.items():
            if param == "verbose" and value:
                solver.verbose = aligator.VERBOSE
            if param == "max_iters":
                solver.max_iters = value
            if param == "default_start" and value:
                xs_init, us_init = default_initialize_rollout(prob)
            if param == "ls_eta":
                solver.linesearch.avg_eta = value

        solver.rollout_type = self._settings.get(
            "rollout_type", aligator.ROLLOUT_NONLINEAR
        )

        bcl_params: solver.AlmParams = solver.bcl_params
        bcl_params.mu_lower_bound = self._settings.get("mu_lower_bound", 1e-10)
        solver.setup(prob)
        results: aligator.Results = solver.results
        try:
            conv = solver.run(prob, xs_init, us_init)
            if conv:
                status = Status.CONVERGED
            elif results.num_iters == solver.max_iters:
                status = Status.MAXITERATIONS
            else:
                status = Status.ERROR
        except Exception:
            status = Status.ERROR
        self._solver = solver
        return Result(
            status.name,
            results.traj_cost,
            results.num_iters,
            results.primal_infeas,
            results.dual_infeas,
        )

    @staticmethod
    def name():
        return "ProxDDP"

    @property
    def solver(self):
        return self._solver


class AltroRunner:
    def __init__(self, settings={}):
        self._solver = None
        self._settings = settings

    @staticmethod
    def name():
        return "ALTRO"

    def solve(self, example, tol: float) -> Result:
        from aligator_bench_pywrap import (
            AltroVerbosity,
            AltroOptions,
            SolveStatus,
            ErrorCodes,
            initAltroFromAligatorProblem,
        )

        p: TrajOptProblem = example.problem
        altro_solver = initAltroFromAligatorProblem(p)
        init_code = altro_solver.Initialize()
        assert init_code == ErrorCodes.NoError

        altro_opts: AltroOptions = altro_solver.GetOptions()
        altro_opts.tol_cost = 1e-16
        altro_opts.tol_primal_feasibility = tol
        # this is actually both the outer loop AND
        # final tolerance...
        altro_opts.iterations_max = 400
        altro_opts.use_backtracking_linesearch = True

        for param, value in self._settings.items():
            if param == "verbose" and value:
                altro_opts.verbose = AltroVerbosity.Inner
            if param == "max_iters":
                altro_opts.iterations_max = value
            if param == "tol_stationarity":
                altro_opts.tol_stationarity = value
            if param == "mu_init":
                altro_opts.penalty_initial = 1 / value

        solver_code = altro_solver.Solve()
        match solver_code:
            case SolveStatus.Success:
                status = Status.CONVERGED
            case SolveStatus.MaxIterations:
                status = Status.MAXITERATIONS
            case _:
                status = Status.ERROR
        print("Solver code:", solver_code)
        self._solver = altro_solver
        return Result(
            status.name,
            altro_solver.CalcCost(),
            altro_solver.GetIterations(),
            altro_solver.GetPrimalFeasibility(),
            altro_solver.GetStationarity(),
        )

    @property
    def solver(self):
        return self._solver


class IpoptRunner:
    def __init__(self, settings={}):
        self._solver = None
        self._settings = settings

    @staticmethod
    def name():
        return "Ipopt"

    def solve(self, example, tol: float) -> Result:
        from aligator_bench_pywrap import SolverIpopt, IpoptApplicationReturnStatus

        self._solver = solver = SolverIpopt()
        p: TrajOptProblem = example.problem
        solver.setup(p)
        solver.setAbsTol(tol)
        for param, value in self._settings.items():
            if param == "max_iters":
                solver.setMaxIters(value)
            if param == "print_level":
                solver.setPrintLevel(value)
            if param == "default_start" and value:
                xs, us = default_initialize_rollout(p)
                solver.setInitialGuess(xs, us)

        solver_code = solver.solve()
        print("Ipopt status:", solver_code)
        match solver_code:
            case IpoptApplicationReturnStatus.Solve_Succeeded:
                status = Status.CONVERGED
            case IpoptApplicationReturnStatus.Maximum_Iterations_Exceeded:
                status = Status.MAXITERATIONS
            case _:
                status = Status.ERROR
        return Result(
            status.name,
            solver.traj_cost,
            solver.num_iter,
            max(solver.cstr_violation, solver.complementarity),
            solver.dual_infeas,
        )

    @property
    def solver(self):
        return self._solver
