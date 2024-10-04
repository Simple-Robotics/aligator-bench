import aligator_bench_pywrap

from aligator import TrajOptProblem


class Result:
    def __init__(self, traj_cost, niter):
        self.traj_cost = traj_cost
        self.niter = niter


class ProxDdpRunner:
    def __init__(self, settings={}):
        self._settings = settings

    def solve(self, example, tol: float):
        p: TrajOptProblem = example.problem


class AltroRunner:
    def __init__(self, settings={}):
        self._solver = None
        self._settings = settings

    def solve(self, example, tol: float):
        from aligator_bench_pywrap import (
            AltroVerbosity,
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

        solver_code = altro_solver.Solve()
        print("Solver code:", solver_code)
        self._solver = altro_solver
        return Result(altro_solver.GetFinalObjective(), altro_solver.GetIterations())

    @property
    def solver(self):
        return self._solver
