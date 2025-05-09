from math import inf
import re
import time
from tracemalloc import start
from pysat.card import CardEnc
from pysat.formula import CNF
from pysat.solvers import Glucose3
import fileinput
from tabulate import tabulate
import webbrowser
import sys
import test

# input variables in database ?? mertens 1
n = 7
m = 2
c = 18
val = 0
cons = 0
sol = 0
solbb = 0
type = 2

toposort = []
clauses = []
time_list = [1, 5, 4, 3, 5, 6, 5]
adj = []
# W = [41, 13, 21, 24, 11, 11, 41, 32, 31, 25, 29, 25, 31, 3, 14, 37, 34, 6, 18, 35, 18, 19, 25, 40, 20, 20, 36, 23, 29, 48, 41, 20, 31, 25, 1]
W = [41, 21, 49, 23, 41, 17, 13]

this_solution = []
def get_this_solution(bestSolution, X, S, A):
    this_solution = [[0 for _ in range(len(A[0]))] for _ in range(len(X[0]))]
    for i in range(len(X[0])):
        for j in range(len(A[0])):
            for k in range(len(X)):
                if bestSolution[S[-1][-1] + k*len(A[0]) + j] > 0 and bestSolution[k*len(X[0]) + i] > 0:
                    this_solution[i][j] = W[int(k)]
    return this_solution
def generate_variables(n,m,c):
    return [[j*m+i+1 for i in range (m)] for j in range(n)], [[m*n + j*c + i + 1 for i in range (c)] for j in range(n)], [[m*n + c*n + j*c + i + 1 for i in range (c)] for j in range(n)]

def generate_extra_variables(k, n, start):
    return [[i*k + j + 1 + start for j in range(k)] for i in range(n - 2)]

def generate_clauses(n,m,c,time_list,adj):
    # #test
    # clauses.append([X[11 - 1][2 - 1]])
    # 1
    for j in range (0, n):
        constraint = []
        for k in range (0, m):
            constraint.append(X[j][k])
        clauses.append(constraint)
    # 2 
    for j in range(0,n):
        for k1 in range (0,m-1):
            for k2 in range(k1+1,m):
                clauses.append([-X[j][k1], -X[j][k2]])

    #3
    for a,b in adj:
        for k in range (0, m-1):
            for h in range(k+1, m):
                clauses.append([-X[b][k], -X[a][h]])
    print("first 3 constraints:", len(clauses))

    #4

    for j in range(n):
        clauses.append([S[j][t] for t in range (c-time_list[j]+1)])

    #5
    for j in range(n):
        for k in range(c-time_list[j]):
            for h in range(k+1, c-time_list[j]+1):
                clauses.append([-S[j][k], -S[j][h]])

    #6
    for j in range(n):
        for t in range(c-time_list[j]+1,c):
            if t > c- time_list[j]:
                clauses.append([-S[j][t]])
    print("4 5 6 constraints:", len(clauses))

    #7
    for i in range(n-1):
        for j in range(i+1,n):
            for k in range (m):
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
    
    #9
    for i, j in adj:
        for k in range(m):
            for t1 in range(c - time_list[i] +1):
                #t1>t2
                for t2 in range(c-time_list[j]+1):
                    if t1 > t2:
                        clauses.append([-X[i][k], -X[j][k], -S[i][t1], -S[j][t2]])
    cons = len(clauses)
    print("Constraints:",cons)
    
    #12 
    for j in range(n):
        if(time_list[j] >= c/2):
            for t in range(c-time_list[j],time_list[j]):
                clauses.append([A[j][t]])
    print("12 constraints:", len(clauses))
    return clauses

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

def solve(solver):
    if solver.solve():
        model = solver.get_model()
        return model
    else:
        # print("no solution")
        return None

def get_value(solution,best_value):
    if solution is None:
        return 100, []
    else:
        x = [[  solution[j*m+i] for i in range (m)] for j in range(n)]
        s = [[  solution[m*n + j*c + i ] for i in range (c)] for j in range(n)]
        a = [[  solution[m*n + c*n + j*c + i ] for i in range (c)] for j in range(n)]
        t = 0
        value = 0

        for t in range(c):
            tmp = 0
            for j in range(n):
                if a[j][t] > 0 :
                    # tmp = tmp + W[j]
                    for l in range(max(0,t-time_list[j]),t+1):
                        if s[j][l] > 0:
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
                        if s[j][l] > 0:
                            tmp = tmp + W[j]
                            station.append(j+1)
                            break
            if tmp >= min(best_value, value):
                constraints.append(station)
                # print("value:",value)
        unique_constraints = list(map(list, set(map(tuple, constraints))))

        return value, unique_constraints

def optimal(X,S,A,n,m,c,sol,solbb,start_time):
    start_time = time.time()
    # print(ip2[])
    clauses = generate_clauses(n,m,c,time_list,adj)

    solver = Glucose3()
    for clause in clauses:
        solver.add_clause(clause)

    model = solve(solver)
    if model is None:
        return None
    bestSolution = model 
    infinity = 1000000
    result = get_value(model, infinity)

    bestValue, station = result
    print("initial value:",bestValue)
    print("initial station:",station)

    this_solution = get_this_solution(model, X, S, A)
    #for row in this_solution:
    #    print(row)
        
    start = A[-1][-1]
    list_time_peak = []
    for t in range(c):
        list_variables = [-1]
        for j in range(len(W)):
            list_variables += [A[j][t]]*W[j]
        list_time_peak.append(list_variables)
    add_con = 0
    for t in range(c):
        # for stations in station: 
        clauses_ = CardEnc.atmost(list_time_peak[t], bestValue -1, 1)
        for clause in clauses_.clauses:
            add_con += 1
            solver.add_clause(clause)
    print("Add cons: ",add_con)
    sol = 1
    solbb = 1
    while True:
        # start_time = time.time()
        sol = sol + 1
        model = solve(solver)
        current_time = time.time()
        if current_time - start_time >= 3600:
            print("time out")
            return bestSolution, sol, solbb, bestValue
        # print(f"Time taken: {end_time - start_time} seconds")
        if model is None:
            return bestSolution, sol, solbb, bestValue
        value, station = get_value(model, bestValue)
        print("value:",value)
        print("Time: ",current_time - start_time)
        # print("station:",station)
        
        this_solution = get_this_solution(model, X, S, A)
        # for row in this_solution:
        #    print(row)

        start = A[-1][-1]
        solver = Glucose3()
        for clause in clauses:
            solver.add_clause(clause)
        for t in range(c):
            # for stations in station: 
            for t in range(c):
                # for stations in station: 
                clauses_ = CardEnc.atmost(list_time_peak[t], value - 1, 1)
                for clause in clauses_.clauses:
                    add_con += 1
                    solver.add_clause(clause)
                # solver.add_clause([-A[j-1][t] for j in stations])
                # print(stations)
        print("Var: ", start)
        print("Add cons: ",add_con)
X, S, A = generate_variables(n,m,c)
val = max(A)
# print(val)
start_time = time.time()
sol = 0
solbb = 0
solution, sol, solbb, solval = optimal(X,S,A,n,m,c,sol,solbb,start_time) #type: ignore
end_time = time.time()
if(solution is not None):
    x = [[solution[j*m+i] for i in range(m)] for j in range(n)]
    s = [[solution[m*n + j*c + i] for i in range(c)] for j in range(n)]
    a = [[solution[m*n + c*n + j*c + i] for i in range(c)] for j in range(n)]
print("Sol: ",sol, ", solbb: ",solbb)
# print(clauses)
# tmp = 0
# for i in time_list: 
#     tmp = tmp + i
# print(tmp)

