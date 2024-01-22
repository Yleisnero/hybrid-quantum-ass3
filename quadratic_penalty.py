from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import LinearEqualityToPenalty, LinearInequalityToPenalty, InequalityToEquality, \
    IntegerToBinary

from instance import example_instance
from quadratic_simple import quadratic_simple


def quadratic_penalty(instance):
    # Extract instance data
    weights, C, B, items = instance

    # Create a Quadratic Program
    qp = QuadraticProgram()

    # Add binary variables x_{ij} for each item i and bin j
    # x_{ij} = 1 if item i is in bin j, else 0

    for j in range(B):
        qp.binary_var(name=f'B_{j}')

    for i in range(items):
        for j in range(B):
            qp.binary_var(name=f'x_{i}_{j}')

    # Objective function: Minimize the number of bins used
    # This is done by minimizing the sum of B_j, where B_j = 1 if bin j is used
    qp.minimize(linear=[1] * B)

    eq_to_pen = LinearEqualityToPenalty(100)
    # Constraint g(x): Each item i must be in exactly one bin
    for i in range(items):
        qp.linear_constraint(
            linear={f'x_{i}_{j}': 1 for j in range(B)},
            sense='==',
            rhs=1,
            name=f'item_in_one_bin_{i}')
        qp = eq_to_pen.convert(qp)
    neq_to_eq = InequalityToEquality("integer")
    eq_to_pen2 = LinearEqualityToPenalty(200)
    int_to_bin = IntegerToBinary()
    # Constraint h(x): Total weight in each bin j must not exceed its capacity C
    for j in range(B):
        # Create a dictionary for the linear terms
        linear_terms = {f'x_{i}_{j}': weights[i] for i in range(items)}
        # Add the term for B_j with a negative coefficient
        linear_terms[f'B_{j}'] = -C
        qp.linear_constraint(
            linear=linear_terms,
            sense='<=',
            rhs=0,  # Set the RHS to 0
            name=f'bin_capacity_{j}')
        qp = neq_to_eq.convert(qp)
        qp = int_to_bin.convert(qp)
        qp = eq_to_pen2.convert(qp)

    return qp


if __name__ == '__main__':
    bpp_quadratic_program = quadratic_penalty(example_instance())
    # Display the quadratic program
    print(bpp_quadratic_program.prettyprint())
    bpp_simple_quadratic_program = quadratic_simple(example_instance())
    print(bpp_simple_quadratic_program.prettyprint())
