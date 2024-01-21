from qiskit import Aer
from qiskit.primitives import Sampler
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA
from qiskit_optimization.algorithms import WarmStartQAOAOptimizer, CplexOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.primitives import BackendSampler

from instance import example_instance
from quadratic_simple import quadratic_simple
from qp_to_qubo import qp_to_qubo
from quadratic_trivial import quadratic_trivial


def solve_qaoa(qubo):
    backend = Aer.get_backend('aer_simulator')
    algorithm_globals.random_seed = 10598
    qaoa_mes = QAOA(sampler=BackendSampler(backend), optimizer=COBYLA(), initial_point=[0.0, 0.0])
    qaoa = MinimumEigenOptimizer(qaoa_mes)
    return qaoa.solve(qubo)


def solve_warm_start_qaoa(qubo):
    algorithm_globals.random_seed = 10598
    qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA(), initial_point=[0.0, 0.0])
    qaoa = WarmStartQAOAOptimizer(pre_solver=CplexOptimizer(),
                                  relax_for_pre_solver=True,
                                  qaoa=qaoa_mes)
    return qaoa.solve(qubo)


def solve_exact(qubo):
    exact_mes = NumPyMinimumEigensolver()
    exact = MinimumEigenOptimizer(exact_mes)
    return exact.solve(qubo)


if __name__ == "__main__":
    quadratic_simple = quadratic_trivial()
    print(quadratic_simple.prettyprint())
    exact_result = solve_exact(quadratic_simple)
    print(exact_result)
    qaoa_result = solve_qaoa(quadratic_simple)
    print(qaoa_result)
    warm_start_qaoa_result = solve_warm_start_qaoa(quadratic_simple)
    print(warm_start_qaoa_result)
