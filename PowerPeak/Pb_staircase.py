from math import inf
import re
import time
import threading
import signal
from tracemalloc import start
from turtle import st
from numpy import var
from pysat.solvers import Glucose3
import fileinput
from tabulate import tabulate
import webbrowser
import sys
from pysat.pb import PBEnc

# input variables in database ?? mertens 1
n = 29 #58
m = 11 #25
c = 33 #65
val = 0
cons = 0    
sol = 0
solbb = 0
type = 1
#           0              1                2           3           4           5           6           7               8                   9
file = ["MITCHELL.IN2","MERTENS.IN2","BOWMAN.IN2","ROSZIEG.IN2","BUXEY.IN2","HESKIA.IN2","SAWYER.IN2","JAESCHKE.IN2","MANSOOR.IN2",
        "JACKSON.IN2","GUNTHER.IN2", "WARNECKE.IN2"]
#            9          10              11          12          13          14          15          16          17   
filename = file[4]

fileName = filename.split(".")

with open('task_power/'+fileName[0]+'.txt', 'r') as file:
    W = [int(line.strip()) for line in file]

neighbors = [[ 0 for i in range(n)] for j in range(n)]
reversed_neighbors = [[ 0 for i in range(n)] for j in range(n)]

visited = [False for i in range(n)]
toposort = []
clauses = []
time_list = []
adj = []
forward = [0 for i in range(n)]
# W = [41, 13, 21, 24, 11, 11, 41, 32, 31, 25, 29, 25, 31, 3, 14, 37, 34, 6, 18, 35, 18, 19, 25, 40, 20, 20, 36, 23, 29, 48, 41, 20, 31, 25, 1]

def input():
    cnt = 0
    for line in fileinput.input('data/'+filename):
        line = line.strip()
        if line:
            if cnt == 0:
                n = int(line)
            elif cnt <= n: # type: ignore
                time_list.append(int(line))
            else:
                line = line.split(",")
                if(line[0] != "-1" and line[1] != "-1"):
                    adj.append([int(line[0])-1, int(line[1])-1])
                    neighbors[int(line[0])-1][int(line[1])-1] = 1
                    reversed_neighbors[int(line[1])-1][int(line[0])-1] = 1
                else:
                    break
            cnt = cnt + 1


def generate_variables(n,m,c):
    x = [[j*m+i+1 for i in range (m)] for j in range(n)]
    a = [[m*n + j*c + i + 1 for i in range (c)] for j in range(n)]
    s = []
    cnt = m*n + c*n + 1
    for j in range(n):
        tmp = []
        for i in range(c - time_list[j] + 1):
            tmp.append(cnt)
            cnt = cnt + 1
        s.append(tmp)
    return x, a, s

def dfs(v):
    visited[v] = True
    for i in range(n):
        if(neighbors[v][i] == 1 and visited[i] == False):
            dfs(i)
    toposort.append(v)
def preprocess(n,m,c,time_list,adj):
    earliest_start = [[-9999999 for _ in range(m)] for _ in range(n)]
    latest_start = [[99999999 for _ in range(m)] for _ in range(n)]
    ip1 = [[0 for _ in range(m)] for _ in range(n)]
    test_ip1 = [[0 for _ in range(m)] for _ in range(n)]
    ip2 = [[[0 for _ in range(c)] for _ in range(m)] for _ in range(n)]
    # Compute earliest possible starting date and assigned workstation
    for i in range(n):
        if not visited[i]:
            dfs(i)
    toposort.reverse()
    # print(toposort)
    for j in toposort:
        k = 0
        earliest_start[j][k] = 0
        for i in range(n):
            if neighbors[i][j] == 1:

                earliest_start[j][k] = max(earliest_start[j][k], earliest_start[i][k] + time_list[i])

                while(earliest_start[j][k] > c - time_list[j]):
                    ip1[j][k] = 1
                    # print('X '+str(j+1)+' '+str(k+1))
                    k = k + 1
                    earliest_start[j][k] = max(0, earliest_start[i][k] + time_list[i])

                if earliest_start[j][k] <= c - time_list[j] :
                    for t in range(earliest_start[j][k]):
                        
                        if(ip2[j][k][t] == 0):
                            # with open("output.txt", "a") as output_file: 
                            #     sys.stdout = output_file  
                            #     print(j+1, k+1, t, file=output_file) 
                            ip2[j][k][t] = 1
    toposort.reverse()
    for j in toposort:
        k = m-1
        latest_start[j][k] = c - time_list[j]
        for i in range(n):
            if(neighbors[j][i] == 1): 
                latest_start[j][k] = min(latest_start[j][k], latest_start[i][k] - time_list[j])
                while(latest_start[j][k] < 0):
                    ip1[j][k] = 1
                    # print('X '+str(j+1)+' '+str(k+1))
                    k = k - 1
                    latest_start[j][k] = min(c - time_list[j], latest_start[i][k] - time_list[j])
                
                if(latest_start[j][k] >= 0):
                        for t in range(latest_start[j][k] + 1, c):
                            
                            if(ip2[j][k][t] == 0):
                                # with open("output.txt", "a") as output_file: 
                                #     sys.stdout = output_file
                                #     print(j+1, k+1, t, file=output_file) 
                                ip2[j][k][t] = 1
    
    # sys.stdout = sys.__stdout__


    # for j in range(n):
    #     for k in range(m):
    #         for t in range(c):
                # if(ip1[j][k] == 1):
                #     continue
                # if(j == 11 or j == 14):
                #     print(f"task {j+1} in machine {k+1} time {t+1}: {ip2[j][k][t]}")
                # if(j == 0 and k == 2):
                #     print(f"task {j+1} in machine {k+1} time {t+1}: {ip2[j][k][t]}")
    # print(ip2)
    return ip1,ip2

def get_key(value):
    for key, value in var_map.items():
        if val == value:
            return key
    return None
def get_var(name, *args):
    global var_counter
    key = (name,) + args

    if key not in var_map:
        var_counter += 1
        var_map[key] = var_counter
    return var_map[key]

def set_var(var, name, *args):
    key = (name,) + args
    if key not in var_map:
        var_map[key] = var
    return var_map[key]

def generate_clauses(n,m,c,time_list,adj,ip1,ip2):
    # #test
    # clauses.append([X[11 - 1][2 - 1]])

    #staircase constraints
    for j in range(n):
        
        set_var(X[j][0], "R", j, 0)
        for k in range(1,m-1):
            if ip1[j][k] == 1:
                set_var(get_var("R", j, k-1), "R", j, k)
            else:
                clauses.append([-get_var("R", j, k-1), get_var("R", j, k)])
                clauses.append([-X[j][k], get_var("R", j, k)])
                clauses.append([-X[j][k], -get_var("R", j, k-1)])
                clauses.append([X[j][k], get_var("R", j, k-1), -get_var("R", j, k)])
        # last machine
        if ip1[j][m-1] == 1:
            clauses.append([get_var("R", j, m-2)])
        else:
            clauses.append([get_var("R", j, m-2), X[j][m-1]])
            clauses.append([-get_var("R", j, m-2), -X[j][m-1]])
        

    for (i,j) in adj:
        for k in range(m-1):
            if ip1[i][k+1] == 1:
                continue
            clauses.append([-get_var("R", j, k), -X[i][k+1]])
    # # 1
    # for j in range (0, n):
    #     # if(forward[j] == 1):
    #     #     continue
    #     constraint = []
    #     for k in range (0, m):
    #         if ip1[j][k] == 1:
    #             continue
    #         constraint.append(X[j][k])
    #     clauses.append(constraint)
    # # 2 
    # for j in range(0,n):
    #     # if(forward[j] == 1):
    #     #     continue
    #     for k1 in range (0,m-1):
    #         for k2 in range(k1+1,m):
    #             if ip1[j][k1] == 1 or ip1[j][k2] == 1:
    #                 continue
    #             clauses.append([-X[j][k1], -X[j][k2]])

    # #3
    # for a,b in adj:
    #     for k in range (0, m-1):
    #         for h in range(k+1, m):
    #             if ip1[b][k] == 1 or ip1[a][h] == 1:
    #                 continue
    #             clauses.append([-X[b][k], -X[a][h]])


    print("first 3 constraints (staircase):", len(clauses))

    # T[j][t] represents "task j starts at time t or earlier"
    for j in range(n):
        last_t = c-time_list[j]
        
        # Special case: Full cycle tasks (only one feasible start time: t=0)
        if last_t == 0:
            # Force the task to start at t=0 (equivalent to original constraint #4)
            clauses.append([S[j][0]])
        else:
            # First time slot
            set_var(S[j][0], "T", j, 0)
            
            # Intermediate time slots
            for t in range(1, last_t):
                clauses.append([-get_var("T", j, t-1), get_var("T", j, t)]) # T[j][t-1] -> T[j][t]
                clauses.append([-S[j][t], get_var("T", j, t)]) # S[j][t] -> T[j][t]
                clauses.append([-S[j][t], -get_var("T", j, t-1)]) # S[j][t] -> ¬T[j][t-1]
                clauses.append([S[j][t], get_var("T", j, t-1), -get_var("T", j, t)]) # T[j][t] -> (T[j][t-1] ∨ S[j][t])
            
            # Last time slot (ensures at least one start time)
            clauses.append([get_var("T", j, last_t-1), S[j][last_t]])
            clauses.append([-get_var("T", j, last_t-1), -S[j][last_t]])
    
    # Original constraints #4 and #5 
    # #4
    # for j in range(n):
    #     clauses.append([S[j][t] for t in range (c-time_list[j]+1)])

    # #5
    # for j in range(n):
    #     for k in range(c-time_list[j]):
    #         for h in range(k+1, c-time_list[j]+1):
    #             clauses.append([-S[j][k], -S[j][h]])

    # #6
    # for j in range(n):
    #     for t in range(c-time_list[j]+1,c):
    #         if t > c- time_list[j]:
    #             clauses.append([-S[j][t]])
    
    print("4 5 6 constraints (staircase):", len(clauses))

    #7
    for i in range(n-1):
        for j in range(i+1,n):
            for k in range (m):
                if ip1[i][k] == 1 or ip1[j][k] == 1 :
                    continue
                for t in range(c):
                    # if ip2[i][k][t] == 1 or ip2[j][k][t] == 1:
                    #     continue
                    clauses.append([-X[i][k], -X[j][k], -A[i][t], -A[j][t]])
    print("7 constraints:", len(clauses))
    #8
    for j in range(n):
        for t in range (c-time_list[j]+1):
            for l in range (time_list[j]):
                if(time_list[j] >= c/2 and t+l >= c-time_list[j] and t+l < time_list[j]):
                    continue
                clauses.append([-S[j][t],A[j][t+l]])
    
    print("8 constraints:", len(clauses))

    # addtional constraints
    # a task cant run before its active time

    # for j in range(n):
    #     for t in range (c-time_list[j]+1):
    #         for l in range (t):
    #             if(time_list[j] >= c/2 and l >= c-time_list[j] and l < time_list[j]):
    #                 continue
    #             clauses.append([-S[j][t],-A[j][l]])


    # addtional constraints option 2


    # for j in range(n):
    #     for t in range (c-1): 
    #         if(time_list[j] >= c/2 and t+1 >= c-time_list[j] and t+1 < time_list[j]):
    #             continue
    #         clauses.append([ -A[j][t], A[j][t+1] , S[j][max(0,t-time_list[j]+1)]])
    
    # #9

    for i,j in adj:
        for k in range(m):
            if ip1[i][k] == 1 or ip1[j][k] == 1:
                continue
            left_bound = time_list[i] - 1
            right_bound = c - time_list[j]
            clauses.append([-X[i][k], -X[j][k], -get_var("T", j, left_bound)])
            for t in range (left_bound + 1, right_bound):
                t_i = t - time_list[i]+1
                clauses.append([-X[i][k], -X[j][k], -get_var("T", j, t), -S[i][t_i]])
            for t in range (max(0,right_bound - time_list[i] + 1), c - time_list[i] + 1):
                clauses.append([-X[i][k], -X[j][k], -S[i][t], -get_var("T",j,c-time_list[j]-1)])
    # for i, j in adj:
    #     for k in range(m):
    #         if ip1[i][k] == 1 or ip1[j][k] == 1:
    #             continue
    #         for t1 in range(c - time_list[i] +1):
    #             #t1>t2
    #             for t2 in range(c-time_list[j]+1):
    #                 if ip2[i][k][t1] == 1 or ip2[j][k][t2] == 1:
    #                     continue
    #                 if t1 > t2:
    #                     clauses.append([-X[i][k], -X[j][k], -S[i][t1], -S[j][t2]])
    cons = len(clauses)
    print("Constraints:",cons)

    #10
    for j in range(n):
        for k in range(m):
            if ip1[j][k] == 1:
                clauses.append([-X[j][k]])
                continue
                # print("constraint ", j+1, k+1)
            #11
            for t in range(c - time_list[j] +1):
                if ip2[j][k][t] == 1:
                    clauses.append([-X[j][k], -S[j][t]])
                    # print("constraint ", j+1, k+1, t)
    
    #12 
    for j in range(n):
        if(time_list[j] >= c/2):
            for t in range(c-time_list[j],time_list[j]):
                clauses.append([A[j][t]])
    print("12 constraints:", len(clauses))
    return clauses

def solve(solver, start_time, timeout=3600):
    current_time = time.time()
    remaining_time = timeout - (current_time - start_time)
    
    if remaining_time <= 0:
        print("Timeout reached before calling solver")
        return None
    
    # Use threading to implement timeout for solver.solve()
    result = [None]  # Use list to allow modification in nested function
    exception = [None]
    
    def solve_worker():
        try:
            if solver.solve():
                result[0] = solver.get_model()
            else:
                result[0] = False  # No solution found
        except Exception as e:
            exception[0] = e
    
    # Start solver in separate thread
    solver_thread = threading.Thread(target=solve_worker)
    solver_thread.daemon = True
    solver_thread.start()
    
    # Wait for thread to complete or timeout
    solver_thread.join(timeout=remaining_time)
    
    if solver_thread.is_alive():
        print(f"Solver timeout reached after {remaining_time:.2f} seconds")
        # Note: We can't actually kill the solver thread, but we return None
        # The thread will continue running in background until solver completes
        return None
    
    if exception[0]:
        raise exception[0]
    
    if result[0] is False:
        # print("no solution")
        return None
    
    return result[0]

def print_solution(solution):
    if solution is None:
        # print("No solution found.")
        return None
    else:
        x = [[solution[j*m+i] for i in range(m)] for j in range(n)]
        a = [[solution[m*n + j*c + i] for i in range(c)] for j in range(n)]
        # s = [[solution[m*n + c*n + j*c + i] for i in range(c)] for j in range(n)]
        cnt = m*n + c*n 
        s = []
        for j in range(n):
            tmp = []
            for i in range(c - time_list[j] + 1):
                tmp.append(solution[cnt])
                cnt += 1
            s.append(tmp)
        table = [[0 for t in range(c)] for k in range(m)]
        for k in range(m):
            for t in range(c):
                for j in range(n):
                    if x[j][k] > 0 and a[j][t] > 0 and table[k][t] == 0:
                        for l in range(max(0,t-time_list[j]),t+1):
                            if l < len(s[j]) and s[j][l] > 0:
                                table[k][t] = j+1

        # Generate HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Task Assignment to Machines</title>
            <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                table, th, td {
                    border: 1px solid black;
                }
                th, td {
                    padding: 8px;
                    text-align: center;
                }
                th {
                    background-color: #f2f2f2;
                }
            </style>
        </head>
        <body>
            <h2>Task Assignment to Machines</h2>
            <table>
                <tr>
                    <th>Machine</th>
                    """ + "".join([f"<th>Time {t+1}</th>" for t in range(c)]) + """
                </tr>
        """

        for k in range(m):
            row = f"<tr><td>Machine {k+1}</td>" + "".join([f"<td>{table[k][t]}</td>" if table[k][t] > 0 else "<td></td>" for t in range(c)]) + "</tr>"
            html_content += row

        html_content += """
            </table>
        </body>
        </html>
        """

        # Write HTML content to a file
        file_path = "task_assignment.html"
        with open(file_path, "w") as file:
            file.write(html_content)

        # Open the HTML file in the default web browser
        # webbrowser.open(file_path)

def get_value(solution,best_value):
    if solution is None:
        return 100, []
    else:
        x = [[  solution[j*m+i] for i in range (m)] for j in range(n)]
        a = [[  solution[m*n + j*c + i ] for i in range (c)] for j in range(n)]
        s = []
        cnt = m*n + c*n
        for j in range(n):
            tmp = []
            for i in range(c - time_list[j] + 1):
                tmp.append(solution[cnt])
                cnt += 1
            s.append(tmp)
        t = 0
        value = 0

        for t in range(c):
            tmp = 0
            for j in range(n):
                if a[j][t] > 0 :
                    # tmp = tmp + W[j]
                    for l in range(max(0,t-time_list[j]),t+1):
                        if l < len(s[j]) and s[j][l] > 0 :
                            tmp = tmp + W[j]
                            # print(tmp)
                            break
                
            if tmp > value:
                value = tmp
                # print(value)

        constraints = []
        for t in range(c):
            tmp = 0
            station = []
            for j in range(n):
                if a[j][t] > 0:
                    # tmp = tmp + W[j]
                    # station.append(j+1)
                    for l in range(max(0,t-time_list[j]),t+1):
                        if l < len(s[j]) and s[j][l] > 0:
                            tmp = tmp + W[j]
                            station.append(j+1)
                            break
            if tmp >= min(best_value, value):
                constraints.append(station)
                # print("value:",value)
        unique_constraints = list(map(list, set(map(tuple, constraints))))

        return value, unique_constraints

def optimal(X,S,A,n,m,c,sol,solbb,start_time):
    global var_counter, var_map
    ip1,ip2 = preprocess(n,m,c,time_list,adj)

    # print(ip2[])


    clauses = generate_clauses(n,m,c,time_list,adj,ip1,ip2)

    solver = Glucose3()
    for clause in clauses:
        solver.add_clause(clause)

    model = solve(solver,start_time = start_time)
    if model is None:
        print("No solution found.")
        return None, sol, solbb, float('inf')
    bestSolution = model 
    infinity = 1000000
    result = get_value(model, infinity)

    bestValue, station = result
    print("initial value:",bestValue)
    print("initial station:",station)
    # for t in range(c):
    #     for stations in station:
            
    #         solver.add_clause([-A[j-1][t] for j in stations])

    #∑(i∈{1,...,n}) w_i * A_{i,t} ≤ bestValue  ∀t∈T
    for t in range(c):
        lits = []
        coeffs = []
        for j in range(n):
            lits.append(A[j][t])
            coeffs.append(W[j])
        pb_enc = PBEnc.leq(lits = lits, weights = coeffs, bound = bestValue - 1 , top_id = var_counter)
        if pb_enc.nv > var_counter:
            var_counter = pb_enc.nv + 1
        for clause in pb_enc.clauses:
            solver.add_clause(clause)
    sol = 1
    solbb = 1
    while True:
        # start_time = time.time()
        sol = sol + 1
        lits = []
        coeffs = []
        model = solve(solver,start_time)
        if model is None:
            # print(bestSolution)
            return bestSolution, sol, solbb, bestValue
        solver = Glucose3() 
        for clause in clauses:
            solver.add_clause(clause)
        
        current_time = time.time()
        if current_time - start_time >= 3600:
            print("time out")
            return bestSolution, sol, solbb, bestValue
        # print(f"Time taken: {end_time - start_time} seconds")
        
        value, station = get_value(model, bestValue)
        # print("value:",value)
        # print("station:",station)
        if value < bestValue:
            solbb = sol
            bestSolution = model
            bestValue = value
            print("new value:",bestValue, end = "\r")
            # print("new station:",station)

        for t in range(c):
            lits = []  
            coeffs = []
            for j in range(n):
                lits.append(A[j][t])
                coeffs.append(W[j])
            pb_enc = PBEnc.leq(lits = lits, weights = coeffs, bound = bestValue - 1 , top_id= var_counter + 1)
            if pb_enc.nv > var_counter:
                var_counter = pb_enc.nv + 1
            for clause in pb_enc.clauses:
                solver.add_clause(clause)
                # print(stations)



input()
X, A, S = generate_variables(n,m,c)
val = max(S)

# print(val)
var_counter = max(val)
var_map = {}

sol = 0
solbb = 0
start_time = time.time()
solution, sol, solbb, solval = optimal(X,S,A,n,m,c,sol,solbb,start_time) #type: ignore

end_time = time.time()

with open("output.txt", "a") as output_file: 
    sys.stdout = output_file
    # print(, file=output_file) 
    print(filename,type,file=output_file)
    
    print("#Var:",var_counter,file=output_file)
    print("#Cons:",len(clauses),file=output_file)
    print("value:",solval,file=output_file)
    print("#sol:",sol,file=output_file)
    print("#solbb:",solbb,file=output_file)
    print(f"Time taken: {end_time - start_time} seconds",file=output_file)
    print(" ",file=output_file)




# print(clauses)
# tmp = 0
# for i in time_list: 
#     tmp = tmp + i
# print(tmp)
