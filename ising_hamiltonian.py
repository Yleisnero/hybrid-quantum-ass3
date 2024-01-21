from qiskit_optimization import QuadraticProgram

from instance import example_instance
from quadratic_simple import quadratic_simple
from qp_to_qubo import qp_to_qubo
from quadratic_trivial import quadratic_trivial


def qubo_to_ising(qubo):
    return qubo.to_ising()


def ising_to_qp(op, offset):
    qp = QuadraticProgram()
    qp.from_ising(op, offset)
    return qp


if __name__ == '__main__':
    quadratic_simple = quadratic_trivial()
    print(quadratic_simple.prettyprint())
    qubo_p = qp_to_qubo(quadratic_simple)
    qubo_p.prettyprint()
    print(qubo_p)
    op_p, offset_p = qubo_to_ising(qubo_p)
    print(op_p)
    print(offset_p)
    quadratic_simple_2 = ising_to_qp(op_p, offset_p)
    print(quadratic_simple_2.prettyprint())




