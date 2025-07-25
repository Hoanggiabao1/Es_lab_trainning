from pysat.formula import CNF, WCNF
from pysat.solvers import Solver
from pysat.pb import PBEnc
from pysat.examples.rc2 import RC2
import math
import time
import fileinput
import csv
import sys
import subprocess

n = 7 # Task number
m = 2 # Workstation number
c = 18 # Cycle
file_name = "MERTENS"

def input_file(file_name):
    W = []
    precedence_relations = set()
    Ex_Time = []

    # Đọc file task_power
    with open(f"task_power/{file_name}.txt") as f:
        for line in f:
            W.append(int(line.strip()))

    # Đọc file data
    with open(f"data/{file_name}.IN2") as f:
        lines = f.readlines()

    n = int(lines[0])
    for idx, line in enumerate(lines[1:], start=1):
        line = line.strip()
        if idx > n:
            pair = tuple(map(int, line.split(',')))
            if pair == (-1, -1):
                break
            precedence_relations.add(pair)
        else:
            Ex_Time.append(int(line))

    return n, W, precedence_relations, Ex_Time

def generate_variables(n, m, c, UB):
    # X[i][s]: task i for station s
    X = [[i*m + s + 1 for s in range(m)] for i in range(n)]
    
    # B[i][t]: task i start at time unit t
    B = [[i*c + t + 1 + X[-1][-1] for t in range(c)] for i in range(n)]
    
    # A[i][t]: task i being executed at time unit t
    A = [[i*c + t + 1 + B[-1][-1] for t in range(c)] for i in range(n)]

    # U[j]: peak >= j
    U = [A[-1][-1] + j + 1 for j in range(UB)]

    return X, B, A, U

def caculate_UB_and_LB(n, m, c, W, precedence_realtion, Ex_Time, A, B, X):
    W_sorted = sorted(W)
    W_sorted.reverse()
    LB = max(W)
    UB = sum(W_sorted[i] for i in range(m))
    print("UB:", UB, "LB:", LB)
    return UB, LB

def list_contrain(n, m, c, precedence_relations, Ex_Time, W, A, B, X):
    wcnf = WCNF()

    # Each task is assigned to exactly one station
    for i in range(n):
        clause = [X[i][s] for s in range(m)]
        wcnf.append(clause)
        for s1 in range(m):
            for s2 in range(s1 + 1, m):
                wcnf.append([-X[i][s1], -X[i][s2]])
    
    # Precedence relations between stations
    for (i,j) in precedence_relations:
        for s1 in range(m):
            for s2 in range(s1):  # s2 < s1
                wcnf.append([-X[i - 1][s1], -X[j - 1][s2]])

    # Precedence relations within same station
    for (i, j) in precedence_relations:
        for s in range(m):
            for t1 in range(c):
                for t2 in range(t1): # t2 < t1   
                    wcnf.append([-X[i - 1][s], -X[j - 1][s], -B[i - 1][t1], -B[j - 1][t2]])
    
    # Each task starts exactly once
    for i in range(n):
        clause = [B[i][t] for t in range(c)]
        wcnf.append(clause)
        for t1 in range(c):
            for t2 in range(t1 + 1, c):
                wcnf.append([-B[i][t1], -B[i][t2]])

    # Tasks must start within feasible time windows
    feasible_start_times = []
    for i in range(n):
        feasible_start_times.append(list(range(c - Ex_Time[i] + 1)))
    for i in range(n):
        for t in range(c):
            if t not in feasible_start_times[i]:
                wcnf.append([-B[i][t]])
    
    # Task activation (B_{i,t} -> A_{i,t+ε} for ε ∈ {0, ..., t_i-1})
    for i in range(n):
        for t in feasible_start_times[i]:
            for epsilon in range(Ex_Time[i]):
                wcnf.append([-B[i][t], A[i][t + epsilon]])
           
    # Task execution (A_{i,t} -> X_{i,s} for s ∈ {1, ..., m})
    # Prevent simultaneous execution on same station
    for i in range(n):
        for j in range(i + 1, n):
            for s in range(m):
                for t in range(c):
                    wcnf.append([-X[i][s], -X[j][s], -A[i][t], -A[j][t]])
    
    return wcnf

def list_inaugural_constrain(n, m, c, UB, LB, precedence_relations, Ex_Time, W):
    wcnf = WCNF()
    X, B, A, U = generate_variables(n, m, c, UB)
    
    wcnf = list_contrain(n, m, c, precedence_relations, Ex_Time, W, A, B, X)

    # Soft constraints: minimize power peak
    for j in range(UB):
        # Manually add to soft constraints 
        wcnf.append([-U[j]], 1)
        
    # Ordering of U variables (U_j -> U_{j-1})
    for j in range(1, UB):
        wcnf.append([-U[j], U[j-1]])
    
    # Power consumption constraint using Pseudo-Boolean encoding
    # ∑(i∈{1,...,n}) w_i * A_{i,t} + ∑(j∈{1,...,UB}) -U_j ≤ UB  ∀t∈T
    start = U[-1] + 1  
    for t in range(c):
        # Build the pseudo-boolean constraint for time unit t
        lits = []
        coeffs = []
        # Add power consumption terms: w_i * A_{i,t}
        for i in range(n):
            lits.append(A[i][t])
            coeffs.append(W[i])
        
        # Add negative U_j terms: -U_j
        for j in range(UB):
            lits.append(-U[j])
            coeffs.append(1)
        # Create PB constraint: sum(coeffs[i] * lits[i]) <= UB
        pb_clauses = PBEnc.leq( lits=lits, weights=coeffs, 
                                bound=UB, 
                                top_id=start)
        # Update variable counter for any new variables created by PBEnc
        if pb_clauses.nv > start:
            start = pb_clauses.nv + 1
            
        # Add the encoded clauses to WCNF
        for clause in pb_clauses.clauses:
            wcnf.append(clause)
    
    var = pb_clauses.nv
    return wcnf, var

def list_binary_constrain(n, m, c, UB, LB, precedence_relations, Ex_Time, W):
    wcnf = WCNF()
    X, B, A, U = generate_variables(n, m, c, UB)
    n_bits = int(math.log2(UB)) + 1
    
    wcnf = list_contrain(n, m, c, precedence_relations, Ex_Time, W, A, B, X)

    # Binary representation variables binU_j
    binU = []
    for j in range(n_bits):
        binU.append(A[-1][-1] + j + 1)
    # Soft constraints: minimize power peak with binary representation
    for j in range(n_bits):
        weight = 2**j
        # Manually add to soft constraints with proper weight
        wcnf.append((-binU[j], weight), weight)
    
    start = binU[-1]
    # Constraint 9': Binary lower bound enforcement
    # ∑(j=0 to ⌈log₂ UB⌉) 2^j * binU_j ≥ LB
    lits_lb = []
    coeffs_lb = []
    for j in range(n_bits):
        lits_lb.append(binU[j])
        coeffs_lb.append(2**j)
        
    # Create PB constraint for lower bound: sum >= LB
    pb_clauses_lb = PBEnc.geq(lits=lits_lb, weights=coeffs_lb, bound=LB,
                                top_id=start)
    # Update variable counter
    if pb_clauses_lb.nv > start:
        start = pb_clauses_lb.nv + 1
    
    # Add the encoded clauses to WCNF
    for clause in pb_clauses_lb.clauses:
        wcnf.append(clause)

     # Constraint 11': Power consumption constraint with binary representation
        # ∑(i∈{1,...,n}) w_i * A_{i,t} ≤ ∑(j=0 to ⌈log₂ UB⌉) 2^j * binU_j  ∀t∈T
        
    for t in range(c):
        # Build the pseudo-boolean constraint for time unit t
        lits = []
        coeffs = []
            
        # Add power consumption terms: w_i * A_{i,t} (positive coefficients)
        for i in range(n):
            lits.append(A[i][t])
            coeffs.append(W[i])
            
        # Add negative binary terms: -2^j * binU_j
        for j in range(n_bits):
            lits.append(-binU[j])
            coeffs.append(2**j)
        # Creat UB' = ∑(j=0 to ⌈log₂ UB⌉) 2^j
        upper_bound = sum(2**j for j in range(n_bits))
        # Create PB constraint: sum(power_terms) - sum(binary_terms) <= 0
        # This is equivalent to: sum(power_terms) <= sum(binary_terms)
        pb_clauses = PBEnc.leq(lits=lits, weights=coeffs, bound=upper_bound,
                                 top_id=start)
            
        # Update variable counter
        if pb_clauses.nv > start:
            start = pb_clauses.nv + 1
            
        # Add the encoded clauses to WCNF
        for clause in pb_clauses.clauses:
            wcnf.append(clause)
    var = pb_clauses.nv
    
    return wcnf, var

def get_value(n, m, c, model, W, UB = 0):
    ans_map = [[0 for _ in range(c)] for _ in range(m + 1)]
    start_B = n*m
    start_A = start_B + n*c
    start_U = start_A + n*c
    
    for i in range(m):
        for j in range(c):
            for k in range(n):
                if ((model[k*m  + i] > 0) and model[start_A + k*c + j] > 0):
                    ans_map[i][j] = W[k]
    
    for i in range(c):
        ans_map[m][i] = sum(ans_map[j][i] for j in range(m))
    peak = max(ans_map[m][i] for i in range(c))
    return ans_map, peak

def write_fancy_table_to_csv(ins, n, m, c, val, s_cons, h_cons, peak, status, time, type, build_time, cal_time, filename="Output.csv"):
    with open("Output/" + filename, "a", newline='') as f:
        writer = csv.writer(f)
        row = []
        row.append(ins)
        row.append(str(n))
        row.append(str(m))
        row.append(str(c))
        row.append(str(val))
        row.append(str(s_cons))
        row.append(str(h_cons))
        row.append(str(peak))
        row.append(status)
        row.append(str(time))
        row.append(type)
        row.append(str(build_time))
        row.append(str(cal_time))
        writer.writerow(row)

def write_fancy_table_to_html(matrix, filename="Output.html", input_file_name="", peak=None):
    with open("Output/" + filename, "w", encoding="utf-8") as f:
        # Viết header HTML
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write("<meta charset='utf-8'>\n")
        f.write("<title>Power Table</title>\n")
        f.write("<style>\n")
        f.write("table {border-collapse: collapse;}\n")
        f.write("td, th {border: 1px solid #333; padding: 5px; text-align: right; font-size: 12px;}\n")
        f.write("th {background-color: #f2f2f2;}\n")
        f.write("h2 {text-align: left;}\n")
        f.write("h3 {color: red; text-align: left;}\n")
        f.write("</style>\n")
        f.write("</head>\n<body>\n")

        f.write(f"<h2>{input_file_name}</h2>\n")

        # Bọc div cho cuộn ngang
        f.write("<div style='overflow-x: auto;'>\n")
        f.write("<table>\n")
        
        # Ghi từng dòng dữ liệu
        for i, row in enumerate(matrix):
            if i == len(matrix) - 1:
                prefix = "Power peak"
            else:
                prefix = "Station " + str(i + 1)

            f.write("<tr>\n")
            f.write(f"<td>{prefix}</td>\n")
            for val in row:
                f.write(f"<td>{val}</td>\n")
            f.write("</tr>\n")

        f.write("</table>\n")
        f.write("</div>\n")
        
        # Thêm dòng cuối ghi Power peak nếu có
        f.write(f"<h3>Power peak = {peak}</h3>\n")

        f.write("</body>\n</html>")

def solve_maxsat_internal(wcnf):
    from pysat.solvers import Glucose3
        
        # First try: solve just the hard constraints
    cnf = CNF()
    for clause in wcnf.hard:
        cnf.append(clause)
        
    solver = Glucose3()
    solver.append_formula(cnf)
        
    if solver.solve():
        solution = solver.get_model()
        solver.delete()
        return solution
    else:
        solver.delete()
        return None

def write_wcnf_with_h_prefix(wcnf, var, filename):
    with open(filename, 'w') as f:
        # Calculate statistics
        total_clauses = len(wcnf.hard) + len(wcnf.soft)
            
        # Calculate top weight safely
        if hasattr(wcnf, 'topw') and wcnf.topw:
            top_weight = wcnf.topw + 1
        elif wcnf.soft:
            try:
                # Try to get weights from soft constraints
                weights = []
                for item in wcnf.soft:
                    if isinstance(item, tuple) and len(item) == 2:
                        weights.append(item[1])  # weight is second element
                    else:
                        # Fallback: assume weight is 1 if not a proper tuple
                        weights.append(1)
                top_weight = max(weights) + 1 if weights else 1
            except Exception as e:
                top_weight = 1000  # Safe fallback
        else:
            top_weight = 1
        
        f.write(f"p wcnf {var} {total_clauses} {top_weight}\n")
        # Write hard constraints with 'h' prefix
        for clause in wcnf.hard:
            f.write(str(top_weight)+" ")
            f.write(" ".join(map(str, clause)))
            f.write(" 0\n")
            
        # Write soft constraints with their weights
        for item in wcnf.soft:
            try:
                if len(item) == 2:
                    clause, weight = item
                else:
                    # Fallback: treat as clause with weight 1
                    clause = item[0]
                    weight = 1
                f.write(f"{weight} ")
                f.write(" " + str(clause))
                f.write(" 0\n")
            except Exception as e:
                continue

def solve_new(wcnf, var):
    wcnf_filename = "problem.wcnf"
    write_wcnf_with_h_prefix(wcnf, var, wcnf_filename)
    # Use external MaxSAT solver (tt-open-wbo-inc)
    try:
        result = subprocess.run([
                                'wsl', './MaxHS/build/release/bin/maxhs',
                                '-printSoln',
                                wcnf_filename
                                ], capture_output=True, text=True, timeout=3600)

        # print(f"Solver output:\n{result.stdout}")
        # Parse solver output
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if line.startswith('v '):
                # Extract variable assignments - could be binary string or space-separated
                var_string = line[2:].strip()
                    
                # Check if it's a binary string (all 0s and 1s)
                if var_string and all(c in '01' for c in var_string):
                    # Convert binary string to variable assignments
                    assignment = []
                    for i, bit in enumerate(var_string):
                        if bit == '1':
                            assignment.append(i + 1)  # Variables are 1-indexed, true
                        else:
                            assignment.append(-(i + 1))
                    return assignment
                else:
                    # Handle space-separated format
                    try:
                        assignment = [int(x) for x in var_string.split() if x != '0']
                        return assignment
                    except ValueError:
                        # Fallback: treat as binary string anyway
                        assignment = []
                        for i, bit in enumerate(var_string):
                            if bit == '1':
                                assignment.append(i + 1)
                        return assignment
        return None
    except subprocess.TimeoutExpired: 
        return None
    
def solve_MaxSat_SAML3P(n, m, c, Ex_Time, W, precedence_relations, file_name, input_in):
    X, B, A, U = generate_variables(n, m, c, 0)
    start_time= time.time()
    UB, LB = caculate_UB_and_LB(n, m, c, W, precedence_relations, Ex_Time, A, B, X)
    cal_UB_LB_time = time.time() - start_time
    start_time = time.time()
    wcnf, var = list_inaugural_constrain(n, m, c, UB, LB, precedence_relations, Ex_Time, W)
    build_time = time.time() - start_time
    model = solve_new(wcnf, var)
    done_time = time.time() - start_time + cal_UB_LB_time
    if model is not None:
        print("Best model found:")
        ans_map, peak = get_value(n, m, c, model, W)
        print("Val = ", var)
        print("Time = ", done_time)
        print("Power peak: ", peak)
        write_fancy_table_to_html(ans_map, filename="Output.html", 
                                      input_file_name=(file_name + " " + str(n) + " " + str(m) + " " + str(c)),
                                      peak = peak)
        write_fancy_table_to_csv(file_name, n, m, c, var, 
                                 len(wcnf.soft), len(wcnf.hard), peak, "optimal",
                                 done_time, "Normal", build_time, cal_UB_LB_time, filename="Normal.csv")
    else:
        print("UNSAT")
        write_fancy_table_to_csv(file_name, n, m, c, var, 
                                 len(wcnf.soft), len(wcnf.hard), " ", "Unsat",
                                 done_time, "Normal", build_time, cal_UB_LB_time, filename="Normal.csv")
    
    start_time = time.time()
    wcnf2, var2 = list_binary_constrain(n, m, c, UB, LB, precedence_relations, Ex_Time, W)
    build_time = time.time() - start_time
    model = solve_new(wcnf2, var2)
    done_time = time.time() - start_time + cal_UB_LB_time
    if model is not None:
        done_time = time.time() - start_time + cal_UB_LB_time
        print("Best model found:")
        ans_map, peak = get_value(n, m, c, model, W, UB)
        write_fancy_table_to_html(ans_map, filename="Output.html", 
                                      input_file_name=(file_name + " " + str(n) + " " + str(m) + " " + str(c)),
                                      peak = peak)
        print("Val = ", var2)
        print("Time = ", done_time)
        print("Power peak: ", peak)
        write_fancy_table_to_csv(file_name, n, m, c, var2, 
                                 len(wcnf2.soft), len(wcnf2.hard), peak, "optimal",
                                 done_time, "Binary", build_time, cal_UB_LB_time, filename="Binary.csv")
    else:
        print("UNSAT")
        write_fancy_table_to_csv(file_name, n, m, c, var2, 
                                 len(wcnf2.soft), len(wcnf2.hard), " ", "timeout",
                                 done_time, "Binary", build_time, cal_UB_LB_time, filename="Binary.csv")

file_name = [
    ["MERTENS", 6, 6],      #0
    ["MERTENS", 2, 18],     #1
    ["BOWMAN", 5, 20],      #2
    ["JAESCHKE", 8, 6],     #3
    ["JAESCHKE", 3, 18],    #4
    ["JACKSON", 8, 7],      #5
    ["JACKSON", 3, 21],     #6
    ["MANSOOR", 4, 48],     #7
    ["MANSOOR", 2, 94],     #8
    ["MITCHELL", 8, 14],    #9
    ["MITCHELL", 3, 39],    #10
    ["ROSZIEG", 10, 14],    #11
    ["ROSZIEG", 4, 32],     #12
    ["BUXEY", 14, 25],      #13
    ["BUXEY", 7, 47],       #14
    ["SAWYER", 14, 25],     #15
    ["SAWYER", 7, 47],      #16
    ["GUNTHER", 14, 40],    #17
    ["GUNTHER", 9, 54],     #18
    ["HESKIA", 8, 138],     #19
    ["BUXEY", 8, 41],       #20
    ["ROSZIEG", 6, 25],     #21
    ["BUXEY", 11, 33],      #22
    ["SAWYER", 12, 30],     #23
    ["SAWYER", 8, 41],      #24
    ["GUNTHER", 9, 61],     #25
    ["HESKIA", 3, 342],     #26
    ["HESKIA", 5, 205]      #27
    ]

with open("Output/Normal.csv", "a", newline='') as f:
        writer = csv.writer(f)
        row = ["MaxHS"]
        writer.writerow(row)

with open("Output/Binary.csv", "a", newline='') as f:
        writer = csv.writer(f)
        row = ["MaxHS"]
        writer.writerow(row)

for input_in in file_name[:19]:
    name = input_in[0]
    m = input_in[1]
    c = input_in[2]
    n, W, precedence_relations, Ex_Time = input_file(name)
    solve_MaxSat_SAML3P(n, m, c, Ex_Time, W, precedence_relations, name, input_in)

