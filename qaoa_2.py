from qiskit import Aer
from qiskit.primitives import Sampler
from qiskit.utils import QuantumInstance
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA
from qiskit_optimization.algorithms import WarmStartQAOAOptimizer, CplexOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from instance import example_instance
from quadratic_simple import quadratic_simple
from qp_to_qubo import qp_to_qubo
from quadratic_trivial import quadratic_trivial






if __name__ == "__main__":
    quadratic_simple = quadratic_trivial()
    qubo = qp_to_qubo(quadratic_simple)
    op, off = qubo.to_ising()
    backend = Aer.get_backend('aer_simulator')
    quantum_instance = QuantumInstance(backend)
    qaoa = QAOA(optimizer=COBYLA(maxiter=250), reps=1)
    result = qaoa.compute_minimum_eigenvalue(operator=op, quantum_instance=quantum_instance)

