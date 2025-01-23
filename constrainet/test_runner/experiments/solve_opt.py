import numpy as np
import torch
from sklearn.utils import check_random_state
from typing import Generator, Tuple
import logging
from time import perf_counter
from json import dumps as json_dumps

from .base import AbstractExperiment, AbstractResultsWriter
from ...problems.linear_constraints import LinearConstraintsGenerator
from ...problems.lin_quad import CannotGenerateProblemError
from ...problems import make_constraints_generator, make_problem_generator
from ...layers import make_solver
from ...utils import prepare_arguments
from ...layers.base import AbstractOptSolver
from ...problems.base import AbstractOptProblem
from ..timeout import exit_after, run_function_with_timeout
from .params_iterator import iterate_params
from pathlib import Path


class SolveOptExperiment(AbstractExperiment):
    def gen_problems(self, random_state: int) -> Generator[Tuple[dict, AbstractOptProblem], None, None]:
        assert isinstance(random_state, int)
        for problem_params in iterate_params(self.problems):
            kind = problem_params['kind']
            name = problem_params['name']
            problem_cache_dir = problem_params.get('problem_cache_dir', '__problem_cache')
            max_attempts = problem_params.get('max_attempts', 1)
            constraints_params = problem_params.get('constraints', dict())
            problem_params = problem_params.get('problem', dict())
            generator_name = f'{kind}Generator'
            # LinearOptProblemGenerator

            # prepare RNG
            rng = check_random_state(random_state)
            # use different generators for constraints and problems
            problem_rng = check_random_state(rng.randint(0, np.iinfo(np.int32).max))

            # prepare generators
            constraints_kind = constraints_params.pop('kind', 'LinearConstraints')
            constraints_gen = make_constraints_generator(constraints_kind, rng)
            # constraints_gen = LinearConstraintsGenerator(rng)
            problem_gen = make_problem_generator(generator_name, random_state=problem_rng)

            problem_cache_file = Path(problem_cache_dir) / name / (f'{random_state:04d}.problem.npz')
            if problem_cache_file.exists():
                logging.info(f'Problem cache file exists ({problem_cache_file!r}), loading from it')
                problem = problem_gen.load(problem_cache_file)
                logging.info(f'Loaded')
            else:
                logging.info(f'Problem cache file does not exist ({problem_cache_file!r}), generating')
                problem_cache_file.parent.mkdir(parents=True, exist_ok=True)
                problem = None

            constraints_args = prepare_arguments(constraints_gen.generate, constraints_params)
            problem_args = prepare_arguments(problem_gen.generate, problem_params)

            for i in range(max_attempts):
                if problem is not None:
                    break
                constraints = constraints_gen.generate(**constraints_args)
                try:
                    problem = problem_gen.generate(constraints, **problem_args)
                    problem_gen.save(problem_cache_file, problem)
                    logging.info(f'Generation successfully finished. Saved the generated problem')
                except CannotGenerateProblemError as ex:
                    logging.error(f"{ex!r}")
                    continue

            if problem is not None:
                descriptor = {
                    'problem_seed': random_state,
                    'problem_name': name,
                    'problem_kind': kind,
                    'problem_constraints_args': json_dumps(constraints_args),
                    'problem_args': json_dumps(problem_args),
                }
                yield descriptor, problem
            else:
                logging.error(
                    f"The problem was not neither loaded nor generated in {max_attempts} attempts"
                )

    def gen_solvers(self, random_state: int) -> Generator[Tuple[dict, AbstractOptSolver], None, None]:
        assert isinstance(random_state, int)
        for problem_params in iterate_params(self.solvers):
            kind = problem_params['kind']
            name = problem_params.get('name', '<unnamed_solver>')
            solver_name = f'{kind}'
            number = problem_params.get('number', 1)
            solver_params = problem_params.get('solver', dict())

            # prepare RNG
            rng = check_random_state(random_state)

            # make solvers
            for i in range(number):
                solver = make_solver(solver_name, random_state=rng, params=solver_params)

                descriptor = {
                    'solver_repetition_id': i,
                    'solver_name': name,  # name of the solver with parameters
                    'solver_kind': kind,
                    'solver_params': json_dumps(solver_params),
                }
                yield descriptor, solver

    def run(self, random_state: int, results_writer: AbstractResultsWriter):
        for problem_desc, problem in self.gen_problems(random_state):
            logging.info(f'Problem {problem_desc!r}')
            exact_solution = problem.exact_solution
            exact_solution_loss = None
            if exact_solution is not None:
                exact_solution_loss = problem.loss(torch.tensor(np.array([exact_solution])))[0].item()
            # iterate over solvers
            for solver_desc, solver in self.gen_solvers(random_state):
                logging.info(f'Solver {solver_desc!r}')

                solve_wrapper = exit_after(self.timeout, name=repr(solver_desc))(solver.solve)
                before_pc = perf_counter()
                # status, solution = run_function_with_timeout(
                #     solver.solve,
                #     [problem],
                #     {},
                #     name=repr(solver_desc),
                #     timeout=self.timeout,
                #     cleanup_time=2.0
                # )
                try:
                    solution = solve_wrapper(problem)
                except KeyboardInterrupt:
                    solution = None
                after_pc = perf_counter()
                duration = after_pc - before_pc

                solution_is_nan = np.any(np.isnan(solution)) if solution is not None else True
                solution_ok = solution is not None and not solution_is_nan
                if solution_ok:
                    solution_satisfy_constraints = problem.check_solution(solution)[0]
                    solution_loss = problem.loss(torch.tensor(np.array([solution])))[0].item()
                    solution_distance = None
                    if exact_solution is not None:
                        solution_distance = np.linalg.norm(solution - exact_solution)
                else:
                    solution_satisfy_constraints = False
                    solution_loss = None
                    solution_distance = None
                    if solution_is_nan:
                        logging.error(f'Solution contains NaN: {solution}')
                    else:
                        logging.error(f'Solution is None. Probably it was stopped due to timeout.')

                result = {
                    'random_state': random_state,
                    'success': solution_ok,
                    'result_exact_solution_loss': exact_solution_loss,
                    'result_solution_loss': solution_loss,
                    'result_solution_satisfy_constraints': solution_satisfy_constraints,
                    'result_solution_distance': solution_distance,
                    'duration': duration,
                }

                logging.info(f'Result={result}')

                results_writer.append({
                    **result,
                    **solver_desc,
                    **problem_desc
                })
