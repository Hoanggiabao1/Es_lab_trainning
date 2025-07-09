from pysat.formula import CNF, WCNF
from pysat.solvers import Solver
from pysat.pb import PBEnc
from pysat.examples.rc2 import RC2
import math
import time
import fileinput
import csv

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
    Time_unit = list(range(c))
    # X[i][s]: task i for station s
    X = []
    for i in range(n):
        row = []
        for s in range (m):
            row.append(i*m + s + 1)
        X.append(row)
    
    # B[i][t]: task i start at time unit t
    B = []
    for i in range(n):
        row = []
        for t in Time_unit:
            row.append(i*c + t + 1 + X[-1][-1])
        B.append(row)
    
    # A[i][t]: task i being executed at time unit t
    A = []
    for i in range(n):
        row = []
        for t in Time_unit:
            row.append(i*c + t + 1 + B[-1][-1])
        A.append(row)
    
    # U[j]: peak >= j
    U = []
    for j in range(0, UB):
        U.append(A[-1][-1] + j + 1)
    return X, B, A, U

def caculate_UB_and_LB(n, m, c, W):
    W_sorted = sorted(W)
    UB = sum(W_sorted[n - m:])
    LB = W_sorted[-1]
    return UB, LB

def list_constrain(n, m, c, UB, LB, precedence_relations, Ex_Time, W):
    wcnf = WCNF()
    X, B, A, U = generate_variables(n, m, c, UB)

    # Soft constraints: minimize power peak
    for j in range(UB):
        # Manually add to soft constraints 
        wcnf.append([-U[j]], 1)
    
    # Each task is assigned to exactly one station
    for i in range(n):
        clause = [X[i][s] for s in range(m)]
        wcnf.append(clause)
        for s1 in range(m - 1):
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
        for t1 in range(c - 1):
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
            for epsilon in list(range(Ex_Time[i])):
                if t + epsilon < c:
                    wcnf.append([-B[i][t], A[i][t + epsilon]])
    
    # Prevent simultaneous execution on same station
    for i in range(n):
        for j in range(i + 1, n):
            for s in range(m):
                for t in range(c):
                    wcnf.append([-X[i][s], -X[j][s], -A[i][t], -A[j][t]])

    # Lower bound enforcement
    for j in range(0, LB):
        wcnf.append([U[j]])
        
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
        
    return wcnf

def list_binary_constrain(n, m, c, UB, LB, precedence_relations, Ex_Time, W):
    wcnf = WCNF()
    X, B, A, U = generate_variables(n, m, c, UB)
    n_bits = math.ceil(math.log2(UB + 1))

    # Binary representation variables binU_j
    binU = []
    for j in range(n_bits):
        binU.append(A[-1][-1] + j + 1)
    
    # Soft constraints: minimize power peak with binary representation (Constraint 1')
    for j in range(n_bits):
        weight = 2**j
        # Manually add to soft constraints with proper weight
        wcnf.append([-binU[j]], weight)
    
    # Each task is assigned to exactly one station
    for i in range(n):
        clause = [X[i][s] for s in range(m)]
        wcnf.append(clause)
        for s1 in range(m - 1):
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
        for t1 in range(c - 1):
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
            for epsilon in list(range(Ex_Time[i])):
                if t + epsilon < c:
                    wcnf.append([-B[i][t], A[i][t + epsilon]])
    
    # Prevent simultaneous execution on same station
    for i in range(n):
        for j in range(i + 1, n):
            for s in range(m):
                for t in range(c):
                    wcnf.append([-X[i][s], -X[j][s], -A[i][t], -A[j][t]])

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

    return wcnf

def get_value(n, m, c, model, UB, LB, W):
    ans_map = []
    start_X = 0
    start_B = n*m
    start_A = start_B + n*c
    start_U = start_A + n*c
    end_U = start_U + UB
    time_peak = []
    for i in range(c):
        time_peak.append(0)
    
    for i in range(m + 1):
        row = []
        for j in range(c):
            row.append(0)
        ans_map.append(row)
    
    for i in range(m):
        for j in range(c):
            for l in range(n):
                if ((model[l*m + i] > 0 and model[start_A + l*c + j] > 0) or 
                    (model[l*m + i] > 0 and model[start_B + l*c + j] > 0)):
                    ans_map[i][j] = W[l]
    
    for i in range(c):
        ans_map[m][i] = sum(ans_map[j][i] for j in range(m))
    peak = max(ans_map[m][i] for i in range(c))
    return ans_map, peak

def write_fancy_table_to_csv(matrix, filename="Output.csv"):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)

        for i, row in enumerate(matrix):
            if i == len(matrix) - 1:
                prefix = "Power peak"
            else:
                prefix = "Station " + str(i + 1)
            # Gộp prefix vào dòng
            csv_row = [prefix] + row
            writer.writerow(csv_row)

def write_fancy_table_to_html(matrix, filename="Output.html", input_file_name="", peak=None):
    with open(filename, "w", encoding="utf-8") as f:
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

def solve_MaxSat_SAML3P(n, m, c, Ex_Time, W, precedence_relations, file_name):
    UB, LB = caculate_UB_and_LB(n, m, c, W)
    #wcnf = list_constrain(n, m, c, UB, LB, precedence_relations, Ex_Time, W)
    wcnf = list_binary_constrain(n, m, c, UB, LB, precedence_relations, Ex_Time, W)
    start_time = time.time()
    print("Soft cons = ", len(wcnf.soft))
    print("Hard cons = ", len(wcnf.hard))
    with RC2(wcnf) as rc2:
        model = rc2.compute()
        if time.time() - start_time > 3600:
            print("Time out")
        elif model is not None:
            done_time = time.time() - start_time
            print("Best model found:")
            ans_map, peak = get_value(n, m, c, model, UB, LB, W)
            print("Val = ", len(model))
            print("Time = ", done_time)
            print("Power peak: ", peak)
            write_fancy_table_to_csv(ans_map, filename="Output.csv")
            write_fancy_table_to_html(ans_map, filename="Output.html", 
                                      input_file_name=(file_name + " " + str(n) + " " + str(m) + " " + str(c)),
                                      peak = peak)
        else:
            print("UNSAT")

file_name = ["BOWMAN", "BUXEY", "GUNTHER", "HESKIA", "JACKSON", "JAESCHKE", "MANSOOR", "MERTENS", "MITCHELL", "ROSZIEG", "SAWYER"]

while True:
    print("="*100)
    for i in range(len(file_name)):
        print(f"{i+1:>2}. {file_name[i]}")
    i = int(input("Data: "))
    if i == 0:
        break
    else:
        m = int(input("Station: "))
        c = int(input("Cycle: "))
        n, W, precedence_relations, Ex_Time = input_file(file_name[i-1])
        solve_MaxSat_SAML3P(n, m, c, Ex_Time, W, precedence_relations, file_name[i-1])
