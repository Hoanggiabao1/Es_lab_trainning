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
    wcnf =list_contrain(n, m, c, precedence_relations, Ex_Time, W, A, B, X)
    model = solve_new(wcnf)
    _, peak = get_value(n, m, c, model, W)
    UB = peak
    W_sorted = sorted(W)
    LB = 0
    sum_of_time = sum(Ex_Time)
    num_tasks = sum_of_time//c + 1
    for i in range(num_tasks):
        LB += W_sorted[i]
    LB = max(LB, max(W))
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
            for t2 in range(t + Ex_Time[i] + 1, c):
                wcnf.append([-B[i][t], -A[i][t2]])

    for i in range(n):
        for t in range(1, c):
            for t2 in range(t):
                wcnf.append([-B[i][t], -A[i][t2]])
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
    for j in range(LB + 1, UB):
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
        for j in range(LB + 1, UB):
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
        wcnf.append([-binU[j]], weight)
    
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
    #if pb_clauses_lb.nv > start:
    #    start = pb_clauses_lb.nv + 1
    
    # Add the encoded clauses to WCNF
    #for clause in pb_clauses_lb.clauses:
    #    wcnf.append(clause)

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
        pb_clauses = PBEnc.leq(lits=lits, weights=coeffs, bound=UB,
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
    
    dem = 0
    for i in range(m):
        for j in range(c):
            for k in range(n):
                if ((model[k*m  + i] > 0) and model[start_A + k*c + j] > 0):
                    ans_map[i][j] = W[k]
                    dem+=1
    print("Number of tasks assigned:", dem)
    
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

def write_wcnf_with_h_prefix(wcnf, filename):
    with open(filename, 'w') as f:
        # Calculate statistics
        total_clauses = len(wcnf.hard) + len(wcnf.soft)
            
        # Calculate top weight safely
        if hasattr(wcnf, 'topw') and wcnf.topw:
            top_weight = wcnf.topw
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
            
            
        # Write hard constraints with 'h' prefix
        for clause in wcnf.hard:
            f.write("h ")
            f.write(" ".join(map(str, clause)))
            f.write(" 0\n")
            
        # Write soft constraints with their weights
        for item in wcnf.soft:
            try:
                if isinstance(item, tuple) and len(item) == 2:
                    clause, weight = item
                else:
                    # Fallback: treat as clause with weight 1
                    clause = item
                    weight = 1
                    
                f.write(f"{weight} ")
                f.write(" ".join(map(str, clause)))
                f.write(" 0\n")
            except Exception as e:

                continue

def solve_new(wcnf):
    wcnf_filename = "problem.wcnf"
    write_wcnf_with_h_prefix(wcnf, wcnf_filename)
    # Use external MaxSAT solver (tt-open-wbo-inc)
    try:
        result = subprocess.run(['wsl','./tt-open-wbo-inc-Glucose4_1_static', wcnf_filename],
                                  capture_output=True, text=True, timeout=3600)
        #print(f"Solver output:\n{result.stdout}")
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
        model = solve_maxsat_internal(wcnf)
        if model is not None:
            return model    
        return None
    
def solve_MaxSat_SAML3P(n, m, c, Ex_Time, W, precedence_relations, file_name, input_in):
    X, B, A, U = generate_variables(n, m, c, 0)
    start_time= time.time()
    UB, LB = caculate_UB_and_LB(n, m, c, W, precedence_relations, Ex_Time, A, B, X)
    cal_UB_LB_time = time.time() - start_time
    start_time = time.time()
    wcnf, var = list_inaugural_constrain(n, m, c, UB, LB, precedence_relations, Ex_Time, W)
    build_time = time.time() - start_time
    model = solve_new(wcnf)
    if model is not None:
        done_time = time.time() - start_time + cal_UB_LB_time
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
                                 done_time, "Normal", build_time, cal_UB_LB_time)
    else:
        print("UNSAT")
        write_fancy_table_to_csv(file_name, n, m, c, var, 
                                 len(wcnf.soft), len(wcnf.hard), " ", "time out",
                                 ">3600", "Normal", build_time, cal_UB_LB_time)
    
    start_time = time.time()
    wcnf2, var2 = list_binary_constrain(n, m, c, UB, LB, precedence_relations, Ex_Time, W)
    build_time = time.time() - start_time
    model = solve_new(wcnf2)
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
                                 done_time, "Binary", build_time, cal_UB_LB_time)
    else:
        print("UNSAT")
        write_fancy_table_to_csv(file_name, n, m, c, var2, 
                                 len(wcnf2.soft), len(wcnf2.hard), " ", "timeout",
                                 ">3600", "Binary", build_time, cal_UB_LB_time)

file_name = [
    ["MERTENS", 6, 6, 164],      #0
    ["MERTENS", 2, 18, 54],     #1
    ["BOWMAN", 5, 20, 146],      #2
    ["JAESCHKE", 8, 6, 173],     #3
    ["JAESCHKE", 3, 18, 47],    #4
    ["JACKSON", 8, 7, 166],      #5
    ["JACKSON", 3, 21, 57],     #6
    ["MANSOOR", 4, 48, 111],     #7
    ["MANSOOR", 2, 94, 71],     #8
    ["MITCHELL", 8, 14, 225],    #9
    ["MITCHELL", 3, 39, 84],    #10
    ["ROSZIEG", 10, 14, 242],    #11
    ["ROSZIEG", 4, 32, 118],     #12
    ["BUXEY", 14, 25, 315],      #13
    ["BUXEY", 7, 47, 184],       #14
    ["SAWYER", 14, 25, 358],     #15
    ["SAWYER", 7, 47, 192],      #16
    ["GUNTHER", 14, 40, 406],    #17
    ["GUNTHER", 9, 54, 276],     #18
    ["GUNTHER", 9, 61, 222],     #19
    ["ROSZIEG", 6, 25, 161],     #20
    ["BUXEY", 8, 41, 245],       #21
    ["BUXEY", 11, 33, 305],      #22
    ["SAWYER", 8, 41, 223],      #23
    ["SAWYER", 12, 30, 303],     #24
    ["HESKIA", 8, 138, 204],     #25
    ["HESKIA", 3, 342, 118],     #26
    ["HESKIA", 5, 205, 141]      #27
    ]

with open("output.csv", "a", newline='') as f:
        writer = csv.writer(f)
        row = ["Cai tien"]
        writer.writerow(row)

for input_in in file_name[:19]:
    name = input_in[0]
    m = input_in[1]
    c = input_in[2]
    n, W, precedence_relations, Ex_Time = input_file(name)
    print(sum(Ex_Time))
    solve_MaxSat_SAML3P(n, m, c, Ex_Time, W, precedence_relations, name, input_in)

