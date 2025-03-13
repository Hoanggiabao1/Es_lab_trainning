from pysat.solvers import Glucose3

def generate_varibales(numbers):
    return [i + 1 for i in range(numbers)]

def generate_extra_variables(extravariables, amount):
    for i in range(amount):
        extravariables.append(extravariables[len(extravariables) - 1]  + 10)

def get_row(variable, start, k):
    return int((variable - start) / k)

def get_col(variable, start, k):
    return (variable - start) % k

def new_sequential_counter_AMK(variables, extravariables, k):
    amount = (len(variables) - 1) * k

    generate_extra_variables(extravariables, amount)

    