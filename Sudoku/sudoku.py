from pysat.solvers import Glucose3

def generate_variables(r, c, s):
    return [(i*r + j)*c + k + 1 for i in range(r) for j in range(c) for k in range(s)]

def at_most_one(clauses, variables):
    length = len(variables)
    if length == 0:
        return clauses
    for i in range(length - 1):
        for j in range(i + 1, length):
            clauses.append([-variables[i], -variables[j]])

    return clauses

def exactly_one(clauses, variables):
    if not variables:
        return clauses
    clauses.append(variables)
    at_most_one(clauses, variables)
    return clauses

def generate_clauses(r, c, s):
    variables = generate_variables(r, c, s)
    clauses = []
    # Each number must appear exactly once in each row, column, and subgrid
    for i in range(r):
        for j in range(c):
            clauses = exactly_one(clauses, [variables[(i*r + j)*c + k] for k in range(s)])
                
    # Row constraints
    for i in range(r):
        for k in range(s):
            variables_row = [variables[(i*r + j)*c + k] for j in range(c)]
            clauses = exactly_one(clauses, variables_row)

    # Column constraints
    for j in range(c):
        for k in range(s):
            variables_col = [variables[(i*r + j)*c + k] for i in range(r)]
            clauses = exactly_one(clauses, variables_col)

    # Subgrid constraints
    for i in range(0, r, 3):
        for j in range(0, c, 3):
            for k in range(s):
                subgrid_vars = [variables[((i + di)*r + (j + dj))*c + k] for di in range(3) for dj in range(3)]
                clauses = exactly_one(clauses, subgrid_vars)
    
    exactly = {(0,4,2), (0,7,6), (0,8,4), (1, 0, 8), (1, 2, 9), (3, 0, 3), 
               (3, 3, 9), (3, 5, 7), (4, 1, 6), (4, 8, 2), (5,3,8),
               (5,6,5), (6,6,9), (6,7,7), (7,1,4), (7, 4, 1), (8, 3, 5)}
    for (i, j, k) in exactly:
        clauses.append([variables[(i*r + j)*c + k - 1]])
    return clauses

def solve_sudoku(r, c, s):
    clauses = generate_clauses(r, c, s)
    solver = Glucose3()
    for clause in clauses:
        solver.add_clause(clause)
    solution = [0] * (r * c * s)
    if solver.solve():
        model = solver.get_model()
        dem = 0
        return model
    else:
        return None
    
solution = solve_sudoku(9,9,9)

def draw_sudoku(solution):
    ans = [[0 for i in range(9)] for j in range(9)]
    for i in range(9):
        for j in range(9):
            for k in range(9):
                if solution[(i*9 + j)*9 + k] > 0:
                    ans[i][j] = k + 1
    for row in ans:
        
        print("|"," | ".join(str(num) for num in row),"|")

draw_sudoku(solution)