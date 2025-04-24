from pysat.solvers import Glucose3

def generate_varibales(numbers):
    return [-1] + [i for i in range(1, numbers + 1)]

def generate_extra_variables(k, n, start):
    return [[i*k + j + 1 + start for j in range(k)] for i in range(n - 2)]

def new_sequential_counter_ALK(clauses, variables, start, k):

    extravariables = generate_extra_variables(k, len(variables), start)
    
    start = extravariables[-1][-1]
    
    # X(i) -> R(i,1)
    for i in range(1, len(variables) - 1):
        clauses.append([-variables[i], extravariables[i - 1][0]])

    # R(i-1,j) -> R(i,j)
    for i in range(2, len(variables) - 1):
        for j in range(min(i - 1, k)):
            clauses.append([-extravariables[i - 2][j], extravariables[i - 1][j]])
    
    # X(i) ^ R(i - 1, j - 1) -> R(i,j)
    for i in range(2, len(variables) - 1):
        for j in range(1, min(i, k)):
            clauses.append([-variables[i], -extravariables[i - 2][j - 1], extravariables[i - 1][j]])
    
    # -X(i) ^ -R(i-1, j) -> -R(i,j)
    for i in range(2, len(variables) - 1):
        for j in range(min(i - 1, k)):
            clauses.append([variables[i], extravariables[i - 2][j], -extravariables[i - 1][j]])
    
    # -X(i) -> -R(i,i)
    for i in range(1, k + 1):
        clauses.append([variables[i], -extravariables[i - 1][i - 1]])
    
    # -R(i-1, j-1) -> -R(i,j)
    for i in range(2, len(variables) - 1):
        for j in range(1, min(i,k)):
            clauses.append([extravariables[i - 2][j - 1], -extravariables[i - 1][j]])
    
    # R(n-1, k) V (X(n) ^ R(n-1)(k-1))
    clauses.append([extravariables[len(variables) - 3][k - 1], variables[len(variables) - 1]])
    if k - 2 >= 0:
        clauses.append([extravariables[len(variables) - 3][k - 1], extravariables[len(variables) - 3][k - 2]])

    return clauses, start

def new_sequential_counter_AMK(clauses, variables, start, k):
    extravariables = generate_extra_variables(k, len(variables), start)
    
    start = extravariables[-1][-1]

    # X(i) -> R(i,1)
    for i in range(1, len(variables) - 1):
        clauses.append([-variables[i], extravariables[i - 1][0]])

    # R(i-1,j) -> R(i,j)
    for i in range(2, len(variables) - 1):
        for j in range(min(i - 1, k)):
            clauses.append([-extravariables[i - 2][j], extravariables[i - 1][j]])
    
    # X(i) ^ R(i - 1, j - 1) -> R(i,j)
    for i in range(2, len(variables) - 1):
        for j in range(1, min(i, k)):
            clauses.append([-variables[i], -extravariables[i - 2][j - 1], extravariables[i - 1][j]])
    
    # X(i) -> -R(i-1, k)
    for i in range(k + 1, len(variables)):
        clauses.append([-variables[i], -extravariables[i - 2][k - 1]])
    
    return clauses, start

k = 2
n = 5
variables = generate_varibales(n)
variables = [-1,1,1,1,2,2]
start = variables[-1]
clauses = []
clauses.append([1])
#clauses.append([2])
#clauses.append([3])
# clauses, start = new_sequential_counter_ALK(clauses, variables, start, k)
clauses, start = new_sequential_counter_AMK(clauses, variables, start, k)

solver = Glucose3()
for clause in clauses:
    print(clause)
    solver.add_clause(clause)

if solver.solve():
    model = solver.get_model()
    print(model[:5])
else:
    print("can't")

