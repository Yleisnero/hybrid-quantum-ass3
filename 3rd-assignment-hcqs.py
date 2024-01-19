# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hybrid Classic-Quantum Systems - Third Assignment #
#
# You should fill in this Jupyter notebook in order to complete the assignment. Here a small recap of the rules:
#
# * You should **NOT** modify the functions that are already provided in the notebook, unless it is strictly necessary;
# * If you want to modify the given functions, please, provide an explanation of why you had to;
# * You can define new functions, as soon as they are well commented;
# * You can import all libraries you want, as soon as you justify their utilization;
# * You can add new cells, as soon as you do not remove the cells where you are supposed to comment your solution;
# * This is a group assignment. The maximum number of people is 3;
# * Your solution should be commented and accompanied by a small description - you can add additional cells if needed;
# * For any issue and doubt, please do not hesitate to use the forum or to write me an email.

# %% [markdown]
# # Preliminaries #
#
# ## Bin Packing Problem (BPP) ##
# The bin packing problem (BPP) is an optimization problem. The final goal is to fit items of different sizes into a finite number of bins, each of a fixed given capacity, minimizing the number of bins used. The problem has many applications, ranging from logistics, multi-processor scheduling.
#
# Computationally, the problem is NP-hard, and the corresponding decision problem, deciding if items can fit into a specified number of bins, is NP-complete.

# %% [markdown]
# ## Mathematical formulation ##
# The objective function of BPP is to minimize the number of bins used, more formally,
# \begin{align}
#     & {\min}
#     & &  \sum_{j=1}^{K} B_j \\
#     & {\text{subject to}}
#     & & g(x_{ij}) = \sum_{j=1}^{K} x_{ij} = 1 \qquad \forall i \\
#     & & & h(x_{ij}) = \sum_{i=1}^N w_{i}x_{ij} \le C*B_j \qquad \forall j
# \end{align}

# %% [markdown]
# The constraint $g(x)$ implies that each task can be packed into at most one CPU, and constraint $h(x_{ij})$ says that the requirements of each task can not exceed the given $C$ capacity. In this paper, BPP can be considered as a mixture of inequality and equality-constrained optimization problems.

# %% [markdown]
# # Loading BPP instances #
# First of all, we focus on generating instances of BPP to be used for the evaluation. Each instance is structured in the following way:
#
# $[[w_0, w_1, ..., w_n], C, B, items]$, such that:
# * $w_i$ is the weight of item $i$;
# * $C$ is the capacity of each bin;
# * $B$ is the number of bins;
# * $items$ is the number of items
#
# We pre-generated the instances for you and saved them in an attached binary file. You can read them in your code by using the following code:

# %%
import pickle

def read_list(filename):
    # for reading also binary mode is important
    with open(filename, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list


# %% [markdown]
# # Transforming BPP into a QUBO #
# QUBO is a mathematical model that is used to represent combinatorial optimization problems in which the variables are binary $\{0,1\}$ and the objective function with constraints is quadratic. We employ QUBO for BPP because it provides a framework for representing combinatorial optimization problems in a form that can be efficiently solved using quantum computing algorithms designed for QUBO problems, such as VQE or QAOA.
#
# ## Slack Variables + Quadratic Penalty Encoding ##
# Qiskit Optimization provides with QuadraticProgram a very generic and powerful representation for optimization problems. However, usually, optimization algorithms cannot handle all possible types of problems that can be modelled, but only a sub-class. Many available quantum optimization algorithms can handle Quadratic Unconstrained Binary Optimization (QUBO) problems. To do so, first, it is necessary to convert a given optimization problem into a QUBO.
#
# Qiskit Optimization provides converters to achieve this conversion whenever possible. More precisely, Qiskit Optimization provides the following converters: 
# * *InequalityToEquality*: converts inequality constraints into equality constraints with additional slack variables.
# * *IntegerToBinary* : converts integer variables into binary variables and corresponding coefficients. 
# * *LinearEqualityToPenalty* : convert equality constraints into additional terms of the object function. 
# * *QuadraticProgramToQubo* : a wrapper for IntegerToBinary and LinearEqualityToPenalty for convenience.
# More information available at: https://qiskit.org/documentation/stable/0.19/tutorials/optimization/2_converters_for_quadratic_programs.html
#
# ## Penalty-Based Encoding ##
#
# ### Define penalties ###
# To enforce the constraints, we introduce penalty terms to the objective function. These terms penalize solutions that violate the constraints. Firsts, we rewrite constraints $g(x_{ij})$ and $h(x_{ij})$ as follows,
#      \begin{align*}
#         g(x_{ij}) = \sum_{j=1}^{K} x_{ij} - 1 = 0 \qquad \forall i \\
#         h(x_{ij}) = \sum_{i=1}^N w_{i}x_{ij} - C \cdot B_j \le 0 \qquad \forall j \\
#     \end{align*}
#     Second, introduce penalty functions $p_1(\lambda, g(x))$ and $p_2(\beta, h(x))$ with coefficients $\{\lambda, \beta\} \geq 0 $
#     \begin{equation}\label{penalty1}
#         p_1(\lambda, g(x_{ij})) =
#         \begin{cases}
#             0 & \text{if $g(x_{ij}) \leq 0$} \\
#             \lambda g(x_{ij}) & \text{if $g(x_{ij}) > 0$} 
#         \end{cases}
#     \end{equation}
#     \begin{equation}\label{penalty2}
#         p_2(\beta, h(x_{ij})) =
#         \begin{cases}
#             0 & \text{if $h(x_{ij}) \leq 0$} \\
#             \beta h(x_{ij}) & \text{if $h(x_{ij}) > 0$} 
#         \end{cases} 
#     \end{equation}
#     
# In literature, there are multiple methods for penalization such as the exterior penalty function and interior penalty function.
#
# ### Injecting Penalties in the Model ###
# To get the final QUBO form we combine penalties with objective function and get
#
# \begin{equation}
#     \mathcal{F} = \sum_{j=1}^{K} B_j + \sum_{j=1}^{K}p_1(\lambda, g_j) + \sum_{i=1}^{N}p_2(\beta,h_i)
# \end{equation}
#
# ### Examples of penalty functions ###
# * https://www.stat.cmu.edu/~ryantibs/convexopt-F13/scribes/lec16.pdf
# * https://www.rose-hulman.edu/~bryan/lottamath/penalty.pdf

# %% [markdown]
# ## YOUR TASK ##
# * Apply slack + quadratic penalty
# * Apply penalty-based encoding using three functions

# %%
from qiskit_optimization.applications import BinPacking
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import CplexOptimizer


bpp_example = read_list("bpp_instances")[12]
print("BPP instance:", bpp_example)

wj = bpp_example[0] # item weights
Q = bpp_example[1]  # capacity of each bin
n = bpp_example[2]  # num of bins
m = bpp_example[3]  # num of items

bp = BinPacking(wj, Q, n)

# %%
# NON QUBO
print("\n" + "=" * 10 + "\nNON-QUBO\n" + "=" * 10)

qp = bp.to_quadratic_program()

result = CplexOptimizer().solve(qp)
print("Result:", result.__repr__())
print("Interpreted result:", bp.interpret(result))

_ = bp.get_figure(result)

# %%
## QUBO
print("\n" + "=" * 10 + "\nQUBO\n" + "=" * 10)

# TODO
penalty = 1
qp_qubo = QuadraticProgramToQubo(penalty=penalty).convert(qp)

result_qubo = CplexOptimizer().solve(qp_qubo)
print("Result:", result_qubo.__repr__())
print("Interpreted result:", bp.interpret(result_qubo))

# FIXME
_ = bp.get_figure(result_qubo)

# %% [markdown]
# # Encoding QUBO into ISING Hamiltonian #
# The next step is to encode the classical QUBO formulation into a quantum state. There exist techniques to encode classical data into quantum such as basis encoding, amplitude encoding, and angle encoding. In this paper, we apply basis encoding, for which any quantum basis can be chosen, but the common way for doing so is to choose basis $\{-1,1\}$ as follows:
# \begin{align}
#     \begin{split}
#     x_i = \frac{1-z_i}{2}\\
#     z_i*z_j = \sigma_Z^i \otimes \sigma_Z^j \\
#     z_i = \sigma_Z^i
#     \end{split}
# \end{align}
# where $\sigma_Z^i$ denotes the Pauli-Z matrix 
# $\begin{pmatrix}
#   1 & 0\\ 
#   0 & -1
# \end{pmatrix}$ on the $i$-th qubit.
# The eigenvalues of $\text{I} - \frac{Z}{2}$ are $\{-1,1\}$ with corresponding eigenstates $|0\rangle$ and $|1\rangle$. Thus, we switch from Z to $\text{I} - \frac{Z}{2}$, and rewrite the problem as BPP Ising hamiltonian,
# \begin{equation}
#     H_{bpp}= \sum_{j=1}^{K} \sigma_j + \sum_{j=1}^{K}p_1(\lambda, g_j) + \sum_{i=1}^{N}p_2(\beta,h_i)
# \end{equation}
# where $g_j$ and $h_i$ are now functions of $\sigma$.

# %% [markdown]
# ## YOUR TASK: Encoding the QUBO as an Ising Hamiltonian ##

# %%
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA

sampler = Sampler()
optimizer = COBYLA()

qaoa = QAOA(reps=1, sampler=sampler, optimizer=optimizer)

# TODO

# QUBO as an Ising Hamiltonian
#operator, offset = qubo.to_ising()

#result_qaoa = qaoa.compute_minimum_eigenvalue(operator=operator)
#
#print(result_qaoa)
#
#plot_bins(result, wj, n, m, l, simplify=True)
#plt.title("QAOA solution", fontsize=18)

# %% [markdown]
# # Solving BPP instances #
# Solve each problem instances using different methods:
# ## CPLEX optimizer ##
# Classic optimizer, to be used as baseline with all the others.

# %%
from docplex.mp.model import Model
from qiskit_optimization.algorithms import CplexOptimizer

# %% [markdown]
# ## QAOA solution ##
#

# %%
from qiskit.algorithms.minimum_eigensolvers import QAOA

# %% [markdown]
# ## Warm start QAOA ##

# %%
from qiskit_optimization.algorithms import WarmStartQAOAOptimizer

# %% [markdown]
# # Evaluation #

# %% [markdown]
# Evaluate quantum solutions in comparison with classic solution using the following metrics:
# * Running time;
# * Mean square error between quantum optimum and classic optimum;
# * Number of times each algorithm reaches the optimum;
#
# Evaluation should be performed on the following backends:
# * Noiseless simulator (Aer)
# * Noisy simulator (choose a noise model)
# * BONUS: test on AQT simulator (check https://github.com/qiskit-community/qiskit-aqt-provider)
#
# QAOA should be evaluated using different configurations:
# * QAOAAnsatz with $p=[1,5]$ 
# * Optimizers: COBYLA, NelderMead, SLSQP
# * Shots: {200, 400, 600, 800, 1000} (Please note that AQT backend is limited to 200 shots)
# * Optimizers' iteration: {250, 500, 750, 1000}

# %%
