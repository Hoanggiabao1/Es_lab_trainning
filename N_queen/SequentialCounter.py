from pysat.solvers import Glucose3

def generate_variables(n):
    return [[i * n + j + 1 for j in range(n)] for i in range(n)]

def generate_extra_variables(extravariable, amount):
    for i in range(amount):
        extravariable.append(extravariable[len(extravariable) - 1] + 1)

def at_most_one(clauses, variables, extravariables):
    length = len(extravariables)
    amount = len(variables) - 1

    generate_extra_variables(extravariables, amount)

    clauses.append([-variables[0], extravariables[length]])
    
    for i in range(1, len(variables) - 1, 1):
        clauses.append([-variables[i], extravariables[length + i]])
        clauses.append([-extravariables[length + i - 1], extravariables[length + i]])
        clauses.append([-variables[i], -extravariables[length + i - 1]])
    
    clauses.append([-variables[len(variables) - 1], -extravariables[len(extravariables) - 1]])

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


n = 4
solution = solve_n_queens(n)
print_solution(solution)