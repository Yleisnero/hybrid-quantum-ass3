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
from quadratic_penalty import quadratic_penalty


def solve_qaoa(qubo):
    backend = Aer.get_backend('aer_simulator')
    backend.set_options(device='GPU')
    algorithm_globals.random_seed = 10598
    qaoa_mes = QAOA(sampler=BackendSampler(backend), optimizer=COBYLA(), initial_point=[0.0, 0.0])
    qaoa = MinimumEigenOptimizer(qaoa_mes)
    return qaoa.solve(qubo)


def solve_warm_start_qaoa(qubo):
    algorithm_globals.random_seed = 10598
    backend = Aer.get_backend('aer_simulator')
    backend.set_options(device='GPU')
    qaoa_mes = QAOA(sampler=BackendSampler(backend), optimizer=COBYLA(), initial_point=[1, 5])
    qaoa = WarmStartQAOAOptimizer(pre_solver=CplexOptimizer(),
                                  relax_for_pre_solver=False,
                                  qaoa=qaoa_mes)
    return qaoa.solve(qubo)


def solve_exact(qubo):
    exact_mes = NumPyMinimumEigensolver()
    exact = MinimumEigenOptimizer(exact_mes)
    return exact.solve(qubo)


def solve_cplex(qubo):
    cplex = CplexOptimizer()
    return cplex.solve(qubo)


if __name__ == "__main__":
    quadratic_simple = quadratic_penalty(example_instance())
    #print(quadratic_simple.prettyprint())
    #exact_result = solve_cplex(quadratic_simple)
    #print(exact_result)
    # print(quadratic_simple.prettyprint())
    # exact_result = solve_exact(quadratic_simple)
    # print(exact_result)
    # qaoa_result = solve_qaoa(quadratic_simple)
    # print(qaoa_result)
    warm_start_qaoa_result = solve_warm_start_qaoa(quadratic_simple)
    print(warm_start_qaoa_result)
