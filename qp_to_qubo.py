from qiskit_optimization.converters import QuadraticProgramToQubo
from quadratic_simple import quadratic_simple, example_instance


def qp_to_qubo(qp):
    qp2qubo = QuadraticProgramToQubo()
    return qp2qubo.convert(qp)


if __name__ == '__main__':
    quadratic_simple = quadratic_simple(example_instance())
    q = qp_to_qubo(quadratic_simple)

    # Display the quadratic program
    print(q.prettyprint())
