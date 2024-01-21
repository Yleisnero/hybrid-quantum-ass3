from qiskit_optimization.algorithms import CplexOptimizer
from instance import example_instance
from quadratic_constrain_rewrite import quadratic_constrain_rewrite
from quadratic_trivial import quadratic_trivial

if __name__ == '__main__':
    # Example instance: 4 items with weights [2, 5, 4, 7], capacity 10, and 3 bins
    simple_quadratic = quadratic_trivial()
    cplex = CplexOptimizer()
    result = cplex.solve(simple_quadratic)
    print(result)
    quadratic_constrain_rewrite = quadratic_constrain_rewrite(example_instance)
    result2 = cplex.solve(quadratic_constrain_rewrite)
    print(result2)

