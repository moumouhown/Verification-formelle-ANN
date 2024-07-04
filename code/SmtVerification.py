from functools import partial
from z3 import *

from modelingActivationFunc import *
from linearAlgebra import *
import time


def loadProperty(propertyPath):
    # Define the file path
    file_path = propertyPath

    # Read the file and store its content in the variable 'property'
    with open(file_path, 'r') as file:
        property = file.read()

    # Print the content of 'property' to verify
    return property


def verification(inputs,outputs,propertyPath):
    s = SolverFor('LRA')  
    p = eval(loadProperty(propertyPath))
    s.add(p)
    
    start_time = time.time()  # Start time
    
    res = s.check()
    
    end_time = time.time()  # End time
    
    
    print("Result:",res)
    print("Verification time: {:.3f}".format(end_time - start_time))  # Print time difference with three digits after the decimal point

    
    
    if res == sat :
        print("Model:", s.model())
    else: 
        print("Model: None")

    print("Log result:" ,s)
    
    
def verification_robustness(inputs,outputs,propertyPath):
    s = SolverFor('LRA')
    loadTextProp = loadProperty(propertyPath)
    
    lines = loadTextProp.splitlines()
    i = 0
    p = ""
    for line in lines:
        if i == 0:
            eps = float(line.replace("eps=", ""))
        else:
            p = p + "\n" + line  
        i += 1
    
    #eps = epsVal  
    pr = eval(p)
    s.add(pr)
    
    start_time = time.time()  # Start time
    
    res = s.check()
    
    end_time = time.time()  # End time
    
    
    print("Result:",res)
    print("Verification time: {:.3f}".format(end_time - start_time))  # Print time difference with three digits after the decimal point

    
    
    if res == sat :
        print("Model:", s.model())
    else: 
        print("Model: None")

    print("Log result:" ,s)