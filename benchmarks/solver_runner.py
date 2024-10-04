import aligator
from aligator import SolverProxDDP, TrajOptProblem
from enum import Enum, auto


class Status(Enum):
    CONVERGED = auto()
    MAXITERATIONS = auto()
    ERROR = auto()


class Result:
    def __init__(self, status: Status, traj_cost, niter):
        self.status = status
        self.traj_cost = traj_cost
        self.niter = niter

    def todict(self):
        return {"status": self.status, "traj_cost": self.traj_cost, "niter": self.niter}

    def __str__(self):
        return f"Result {self.todict()}"


class ProxDdpRunner:
    def __init__(self, settings={}):
        self._settings = settings

    def solve(self, example, tol: float):
        prob: TrajOptProblem = example.problem
        solver = SolverProxDDP(tol)
        solver.mu_init = self._settings["mu_init"]  # required param
        for param, value in self._settings.items():
            if param == "verbose" and value:
                solver.verbose = aligator.VERBOSE
            if param == "max_iters":
                solver.max_iters = value

        bcl_params: solver.AlmParams = solver.bcl_params
        bcl_params.mu_lower_bound = self._settings.setdefault("mu_lower_bound", 1e-10)
        solver.setup(prob)
        results: aligator.Results = solver.results
        try:
            conv = solver.run(prob)
            if conv:
                status = Status.CONVERGED
            elif results.num_iters == solver.max_iters:
                status = Status.MAXITERATIONS
            else:
                status = Status.ERROR
        except Exception:
            status = Status.ERROR
        self._solver = solver
        return Result(status, results.traj_cost, results.num_iters)

    @property
    def solver(self):
        return self._solver


class AltroRunner:
    def __init__(self, settings={}):
        self._solver = None
        self._settings = settings

    def solve(self, example, tol: float):
        from aligator_bench_pywrap import (
            AltroVerbosity,
            SolveStatus,
            ErrorCodes,
            initAltroFromAligatorProblem,
        )

        p: TrajOptProblem = example.problem
        altro_solver = initAltroFromAligatorProblem(p)
        init_code = altro_solver.Initialize()
        assert init_code == ErrorCodes.NoError

        altro_opts = altro_solver.GetOptions()
        altro_opts.tol_cost = 1e-16
        altro_opts.tol_primal_feasibility = tol
        altro_opts.tol_stationarity = tol
        altro_opts.iterations_max = 400
        altro_opts.use_backtracking_linesearch = True

        for param, value in self._settings.items():
            if param == "verbose" and value:
                altro_opts.verbose = AltroVerbosity.Inner
            if param == "max_iters":
                altro_opts.iterations_max = value

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
        return Result(status, altro_solver.CalcCost(), altro_solver.GetIterations())

    @property
    def solver(self):
        return self._solver
