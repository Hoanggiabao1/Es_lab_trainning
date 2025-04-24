from pysat.solvers import Glucose3

def generate_variables(row, col, start):
    return [[i*col + j + start + 1 for j in range(col)] for i in range (row)]

def generate_scheduling_variables(row, col, start):
    return [[i*len(col) + col[i][j] + start + 1 for j in range(len(col[i]))] for i in range (row)]
def at_least_one(clauses, variables):
    clauses.append(variables)
    return clauses

def at_most_one(clauses, variables):
    for i in range(len(variables) - 1):
        for j in range(i + 1, len(variables)):
            clauses.append([-variables[i], -variables[j]])
    return clauses

def exactly_one(clauses, variables):
    clauses = at_least_one(clauses, variables)
    clauses = at_most_one(clauses, variables)
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

def list_of_contraints(task, rime_slots, time_slots_posible, workstations):
    clauses = []
    # X[i][j]: task i assigned to workstation j
    assignment_variables = generate_variables(len(tasks), len(workstations), 0)

    # S[i][j]: task i start in time slot j
    scheduling_varibales = generate_scheduling_variables(len(tasks), time_slots_posible, assignment_variables[-1][-1])

    # A[i][j]: task i running in time slot j
    activity_variables = generate_variables(len(tasks), len(time_slots), scheduling_varibales[-1][-1])

    # Each task must be assigned to exactly one workstation
    for i in range(len(tasks)):
        clauses = exactly_one(clauses, assignment_variables[i])

    # Each task must be start exactly one time
    for i in range(len(tasks)):
        clauses = exactly_one(clauses, scheduling_varibales[i])
    
    # Not possible to have 2 different tasks running at the same time on the same working station
    for i in range(len(tasks) - 1):
        for j in range(i + 1, len(tasks)):
            for k in range(len(workstations)):
                for t in range(len(time_slots)):
                    clauses.append([-assignment_variables[i][k], -assignment_variables[j][k], 
                                    -activity_variables[i][t], -activity_variables[j][t]])
    
    # Precedence contrains i < j, task i can't be assigned to a workstation whose number
    # exceeds the number of the workstation on which j assigned
    for i in range(len(tasks) - 1):
        for j in range(i + 1, len(tasks)):
            for k in range(len(workstations) - 1):
                for h in range(k + 1, len(workstations)):
                    clauses.append([-assignment_variables[j][k], -assignment_variables[i][h]])
    
    # Precedence constrain i < j, if both tasks are assigned to the same workstation
    # then task i can't start after task j
    for i in range(len(tasks) - 1):
        for j in range(i + 1, len(tasks)):
            for k in range(len(workstations)):
                for t1 in range(len(time_slots_posible[i])):
                    for t2 in range(len(time_slots_posible[j])):
                        if (t1 > t2):
                            clauses.append([-assignment_variables[i][k], -assignment_variables[j][k],
                                            -scheduling_varibales[i][t1], -scheduling_varibales[j][t2]])
    
    # If task duration is more than half of cycle time, this task is necessarily
    # active in the middle of the time horizon
    for i in range(len(tasks)):
        if (len(time_slots) - len(time_slots_posible[i]) < len(time_slots_posible[i]) - 1):
            for t in range(len(time_slots) - len(time_slots_posible[i]), len(time_slots_posible[i]) - 1):
                clauses.append([activity_variables[i][t]])


tasks = [1, 2, 3, 4]
durations = [2, 4, 3, 3]
time_slots = [0, 1, 2, 3, 4]
time_slots_posible = []
for i in range(len(tasks)):
    time_slots_posible.append(time_slots[:len(time_slots) + 1 - durations[i]])
workstations = [1, 2, 3]

assignment_variables = generate_variables(len(tasks), len(workstations), 0)
scheduling_varibales = generate_scheduling_variables(len(tasks), time_slots_posible, assignment_variables[-1][-1])
activity_variables = generate_variables(len(tasks), len(time_slots), scheduling_varibales[-1][-1])

for variable in assignment_variables:
    print(variable)

for scheduling_variable in scheduling_varibales:
    print(scheduling_variable)

for activity_variable in activity_variables:
    print(activity_variable)