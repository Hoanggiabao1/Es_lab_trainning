from pysat.solvers import Glucose3

def generate_varibales(numbers):
    return [-1] + [i for i in range(1, numbers + 1)]

def generate_extra_variables(k, n, start):
    return [[i*k + j + 1 + start for j in range(k)] for i in range(n - 2)]

def at_most_one(clauses, variables):
    for i in range(1, len(variables)):
        for j in range(i + 1, len(variables)):
            clauses.append([-variables[i], -variables[j]])
    return clauses

def at_least_one(clauses, variables):
    clauses.append(variables[1:])
    return clauses

def generate_k_subsets(arr, k, index=0, current=[], result=[]):
    if len(current) == k:
        result.append(current[:])
        return
    
    if index == len(arr):
        return
    
    current.append(arr[index])
    generate_k_subsets(arr, k, index + 1, current, result)
    
    current.pop()
    generate_k_subsets(arr, k, index + 1, current, result)

    return result

def naive_ALK(clauses, variables, k):

    subsets_n_k_1 = []
    subsets_n_k_1 = generate_k_subsets(variables, len(variables) - k + 1, 0, [], subsets_n_k_1)

    for subset in subsets_n_k_1:
        clauses = at_least_one(clauses, subset)

    return clauses

def naive_AMK(clauses, variables, k):

    subsets_k_1 = []
    subsets_k_1 = generate_k_subsets(variables, k + 2, 0, [], subsets_k_1)

    for subset in subsets_k_1:
        not_subset = [-i for i in subset]
        clauses = at_least_one(clauses, not_subset)
    
    return clauses

k = 2
n = 5
variables = generate_varibales(n)
clauses = []

clauses = naive_ALK(clauses, variables, k)
clauses = naive_AMK(clauses, variables, k)

solver = Glucose3()
for clause in clauses:
    solver.add_clause(clause)

if solver.solve():
    model = solver.get_model()
    print(model[:5])
else:
    print("can't")

