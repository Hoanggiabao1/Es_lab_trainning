from pysat.solvers import Glucose3


def generate_variables(n):
    return [[i * n + j + 1 for j in range(n)] for i in range(n)]

def generate_extra_variables(k, n, start):
    return [[i*k + j + 1 + start for j in range(k)] for i in range(n - 1)]

def new_sequential_counter_ALK(clauses, variables, start, k):

    extravariables = generate_extra_variables(k, len(variables), start)
    
    start = extravariables[-1][-1]
    
    # print(extravariables)
    # X(i) -> R(i,1)
    for i in range(len(variables) - 1):
        clauses.append([-variables[i], extravariables[i][0]])

    # R(i-1,j) -> R(i,j)
    for i in range(1, len(variables) - 1):
        for j in range(min(i, k)):
            clauses.append([-extravariables[i - 1][j], extravariables[i][j]])
    
    # X(i) ^ R(i - 1, j - 1) -> R(i,j)
    for i in range(1, len(variables) - 1):
        for j in range(1, min(i + 1, k)):
            clauses.append([-variables[i], -extravariables[i - 1][j - 1], extravariables[i][j]])
    
    # -X(i) ^ -R(i-1, j) -> -R(i,j)
    for i in range(1, len(variables) - 1):
        for j in range(min(i, k)):
            clauses.append([variables[i], extravariables[i - 1][j], -extravariables[i][j]])
    
    # -X(i) -> -R(i,i)
    for i in range(k):
        clauses.append([variables[i], -extravariables[i][i]])
    
    # -R(i-1, j-1) -> -R(i,j)
    for i in range(1, len(variables) - 1):
        for j in range(1, min(i + 1,k)):
            clauses.append([extravariables[i - 1][j - 1], -extravariables[i][j]])
    
    # R(n-1, k) V (X(n) ^ R(n-1)(k-1))
    clauses.append([extravariables[len(variables) - 2][k - 1], variables[len(variables) - 1]])
    if k - 2 >= 0:
        clauses.append([extravariables[len(variables) - 2][k - 1], extravariables[len(variables) - 2][k - 2]])

    return clauses, start

def new_sequential_counter_AMK(clauses, variables, start, k):
    
    extravariables = generate_extra_variables(k, len(variables), start)
    
    start = extravariables[-1][-1]

    # X(i) -> R(i,1)
    for i in range(len(variables) - 1):
        clauses.append([-variables[i], extravariables[i][0]])

    # R(i-1,j) -> R(i,j)
    for i in range(1, len(variables) - 1):
        for j in range(min(i, k)):
            clauses.append([-extravariables[i - 1][j], extravariables[i][j]])
    
    # X(i) ^ R(i - 1, j - 1) -> R(i,j)
    for i in range(1, len(variables) - 1):
        for j in range(1, min(i + 1, k)):
            clauses.append([-variables[i], -extravariables[i - 1][j - 1], extravariables[i][j]])
    
    # X(i) -> -R(i-1, k)
    for i in range(k, len(variables)):
        clauses.append([-variables[i], -extravariables[i - 1][k - 1]])
    
    return clauses, start


def exactly_one(clauses, variables, start):
    clauses, start = new_sequential_counter_ALK(clauses, variables, start, 1)
    clauses, start = new_sequential_counter_AMK(clauses, variables, start, 1)
    return clauses, start

def generate_clauses(n, variables, start):
    clauses = []

    # Each row
    for i in range(n):
        clauses, start = exactly_one(clauses, variables[i], start)

    # Each column
    for j in range(n):
        clauses, start = exactly_one(clauses, [variables[i][j] for i in range(n)], start)


    # Each diagonal
    for i in range(n):
        for j in range(n):
            for k in range(1, n):
                if i + k < n and j + k < n:
                    clauses, start = new_sequential_counter_AMK(clauses, [variables[i][j], variables[i + k][j + k]], start, 1)
                if i + k < n and j - k >= 0:
                    clauses, start = new_sequential_counter_AMK(clauses, [variables[i][j], variables[i + k][j - k]], start, 1)

    return clauses, start


def solve_n_queens(n):
    variables = generate_variables(n)
    start = variables[-1][-1]
    clauses, start = generate_clauses(n, variables, start)

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