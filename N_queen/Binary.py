from pysat.solvers import Glucose3
from itertools import product
import math

def generate_variables(n):
    return [[i * n + j + 1 for j in range(n)] for i in range(n)]

def generate_extra_variables(extravariable, amount):
    for i in range(amount):
        extravariable.append(extravariable[len(extravariable) - 1] + 1)

def at_most_one(clauses, variables, extravariables):
    length = len(extravariables)
    amount = math.ceil(math.log2(len(variables)))

    generate_extra_variables(extravariables, amount)

    # Generate Binary array for variables
    binary_values = [list(row) for row in product([0, 1], repeat=len(variables))]

    for i in range(len(variables)):
        for j in range(length, len(extravariables), 1):
            clauses.append([-variables[i], extravariables[j] * ((-1)**binary_values[i][j - length - 1])])
    return clauses


def exactly_one(clauses, variables, extravariables):
    clauses.append(variables)
    at_most_one(clauses, variables, extravariables)


def generate_clauses(n, variables):
    clauses = []
    extravariables = [variables[-1][-1]]
    # Each row
    for i in range(n):
        exactly_one(clauses, variables[i], extravariables)

    # Each column
    for j in range(n):
        exactly_one(clauses, [variables[i][j] for i in range(n)], extravariables)


    # Each diagonal
    for i in range(n):
        for j in range(n):
            for k in range(1, n):
                if i + k < n and j + k < n:
                    at_most_one(clauses, [variables[i][j], variables[i + k][j + k]], extravariables)
                if i + k < n and j - k >= 0:
                    at_most_one(clauses, [variables[i][j], variables[i + k][j - k]], extravariables)

    return clauses


def solve_n_queens(n):
    variables = generate_variables(n)
    clauses = generate_clauses(n, variables)

    solver = Glucose3()
    for clause in clauses:
        solver.add_clause(clause)

    if solver.solve():
        model = solver.get_model()
        return [[int(model[i * n + j] > 0) for j in range(n)] for i in range(n)]
    else:
        return None


def print_solution(solution):
    if solution is None:
        print("No solution found.")
    else:
        for row in solution:
            print(" ".join("Q" if cell else "." for cell in row))


n = 16
solution = solve_n_queens(n)
print_solution(solution)