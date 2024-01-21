from qiskit_optimization import QuadraticProgram

from instance import example_instance


def quadratic_trivial():
    qp = QuadraticProgram()
    qp.integer_var(name='x', lowerbound=-500, upperbound=1)
    qp.minimize(linear=[1])
    return qp


if __name__ == '__main__':
    bpp_quadratic_program = quadratic_trivial()
    # Display the quadratic program
    print(bpp_quadratic_program.prettyprint())
