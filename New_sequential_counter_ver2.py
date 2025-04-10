from pysat.solvers import Glucose3

def generate_varibales(numbers):
    return [-1] + [i for i in range(1, numbers + 1)]

def generate_extra_variables(k, n, start):
    return [[i*k + j + 1 + start for j in range(k)] for i in range(n - 2)]

def exactly_k(clauses, variables, k):
    

k = 2
n = 5
variables = generate_varibales(n)
start = variables[-1]
clauses = []

solver = Glucose3()
for clause in clauses:
    solver.add_clause(clause)

if solver.solve():
    model = solver.get_model()
    print(model[:5])
else:
    print("can't")

