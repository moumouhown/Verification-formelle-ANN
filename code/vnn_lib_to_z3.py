from z3 import *
import re
def parse_vnnlib_expr(expr, variables):
    """
    Parses a vnnlib expression into a Z3 expression.
    
    :param expr: A vnnlib expression (as a list).
    :param variables: A dictionary mapping variable names to Z3 variables.
    :return: A Z3 expression.
    """
    if isinstance(expr, str):
        if expr in variables:
            return variables[expr]
        elif re.match(r'^-?\d+(\.\d+)?$', expr):  # Match integers and floats
            return float(expr) if '.' in expr else int(expr)
        else:
            raise ValueError(f"Unknown variable or constant: {expr}")
    
    op = expr[0]
    
    if op == 'assert':
        return parse_vnnlib_expr(expr[1], variables)
    elif op == 'AND':
        return And(*[parse_vnnlib_expr(sub_expr, variables) for sub_expr in expr[1:]])
    elif op == 'OR':
        return Or(*[parse_vnnlib_expr(sub_expr, variables) for sub_expr in expr[1:]])
    elif op == '>':
        return parse_vnnlib_expr(expr[1], variables) > parse_vnnlib_expr(expr[2], variables)
    elif op == '<':
        return parse_vnnlib_expr(expr[1], variables) < parse_vnnlib_expr(expr[2], variables)
    elif op == '>=':
        return parse_vnnlib_expr(expr[1], variables) >= parse_vnnlib_expr(expr[2], variables)
    elif op == '<=':
        return parse_vnnlib_expr(expr[1], variables) <= parse_vnnlib_expr(expr[2], variables)
    elif op == '=':
        return parse_vnnlib_expr(expr[1], variables) == parse_vnnlib_expr(expr[2], variables)
    else:
        raise ValueError(f"Unknown operation: {op}")

def vnnlib_to_z3(vnnlib_str):
    """
    Converts a vnnlib constraint string to a Z3 expression.
    
    :param vnnlib_str: A vnnlib constraint string.
    :return: A Z3 expression.
    """
    # Tokenize the input string
    tokens = vnnlib_str.replace('(', ' ( ').replace(')', ' ) ').split()
    # Function to convert the tokens to a nested list (parsed expression)
    def parse_tokens(tokens):
        if not tokens:
            raise ValueError("Unexpected end of tokens")
        token = tokens.pop(0)
        if token == '(':
            sub_expr = []
            while tokens[0] != ')':
                sub_expr.append(parse_tokens(tokens))
            tokens.pop(0)  # remove ')'
            return sub_expr
        elif token == ')':
            raise ValueError("Unexpected ')'")
        else:
            return token
    
    parsed_expr = parse_tokens(tokens)
    
    # Create Z3 variables for all variables in the parsed expression
    def collect_variables(expr, variables):
        if isinstance(expr, str):
            if expr.isalpha() and expr not in variables:
                variables[expr] = Real(expr)
        else:
            for sub_expr in expr:
                collect_variables(sub_expr, variables)
    
    variables = {}
    collect_variables(parsed_expr, variables)
    
    # Parse the vnnlib expression into a Z3 expression
    z3_expr = parse_vnnlib_expr(parsed_expr, variables)
    
    return z3_expr

def vnn_lib_file_to_z3(file_path):
    variables = []
    constraints = []
    output = []
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    # Find the line with "; Declaration of variables of interest"
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        print(line)
        #variable declaration
        if line.strip() == "; Declaration of variables of interest" or line.startswith("(declare-fun"):
            i += 1
            if(line.startswith("(declare-fun")):
                i-=1
            # Start reading (declare-fun x() Real) lines
            while i < len(lines):
                print("here variable")
                line = lines[i].strip()
                if line.startswith("(declare-fun") and "Real" in line:
                    # Extract the variable name and add to the list
                    parts = line.split()
                    var_name = parts[1]
                    print(var_name)
                    variables.append(Real(var_name))
                else:
                    break
                i += 1
        #constraints 
        if line.strip() == "; Constraints" or line.startswith("(assert"):
            i += 1
            if(line.startswith("(assert")):
                i-=1
            # Start reading (assert) lines
            while i < len(lines):
                print("here constraint")
                line = lines[i].strip()
                if line.startswith("(assert") :
                    # Extract the variable name and add to the list
                    constraint = line
                    print(constraint)
                    constraints.append(vnnlib_to_z3(constraint))
                else:
                    break
                i += 1
        #output
        if line.strip() == "; Desired output" or line.startswith("(assert"):
            i += 1
            if(line.startswith("(assert")):
                i-=1
            # Start reading desired output
            while i < len(lines):
                print("here output")
                line = lines[i].strip()
                if line.startswith("(assert") :
                    # Extract the variable name and add to the list
                    constraint = line
                    print(constraint)
                    output.append(vnnlib_to_z3(constraint))
                else:
                    break
                i+=1
        i += 1
    
    return variables, constraints, output

# Example usage
# Read variables of interest from "test.txt"
file_path = "test.txt"
z3_converted_file_path="test_z3.txt"
variables, constraints, output = vnn_lib_file_to_z3(file_path)
#print("Z3 Format:\n")
#print(variables, constraints, output)
with open(z3_converted_file_path, "w") as file:
    file.write(str(variables))
    file.write("\n")
    file.write(str(constraints))
    file.write("\n")
    file.write(str(output))


#Testing part with local data
import z3

# Import logical operators explicitly
from z3 import And, Or

# Function to parse Z3 expressions from the file
def parse_expressions(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return lines

# Load variables and constraints from file
vnn_lib_string = parse_expressions(file_path)

# Convert "AND" and "OR" within lines to lowercase
for i, vnn_lib_str in enumerate(vnn_lib_string):
    if "AND" in vnn_lib_str or "OR" in vnn_lib_str:
        vnn_lib_string[i] = vnn_lib_str.replace("AND", "and").replace("OR", "or")

print("vnn string: ", vnn_lib_string)

# Create a Z3 solver instance
solver = z3.Solver()

for vnn_lib_str in vnn_lib_string:
    print(vnn_lib_str.strip())
    solver.from_string(vnn_lib_str.strip())

# Check satisfiability
result = solver.check()

# Print result
if result == z3.sat:
    print("Satisfiable")
    model = solver.model()
    print("Model:", model)
elif result == z3.unsat:
    print("Unsatisfiable")
else:
    print("Unknown")

#Testing part with z3 file
import z3

# Function to read and parse the file
def parse_z3_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    variables_str = lines[0].strip()
    constraints_str = lines[1].strip()
    desired_output_str = lines[2].strip()
    print("z3 content : ", variables_str, constraints_str, desired_output_str)
    return variables_str, constraints_str, desired_output_str

# Function to create Z3 variables from the variable string
def create_variables(variables_str):
    variables = {}
    # Find all variable names using regex
    variable_names = re.findall(r'\b\w+\b(?=\(\))', variables_str)
    # Generate Z3 variable declarations
    variables_exec_str = '\n'.join([f"{var} = z3.Real('{var}')" for var in variable_names])
    print("variables_exec_str: ",variables_exec_str)
    exec(variables_exec_str, {'z3': z3}, variables)
    return variables

# Function to create Z3 constraints from the constraint string
def create_constraints(constraints_str, variables):
    local_dict = {}
    exec(f"from z3 import And, Or\nconstraints = {constraints_str}", variables, local_dict)
    return local_dict['constraints']

# Function to create Z3 constraint from the desired output string
def create_desired_output(desired_output_str, variables):
    local_dict = {}
    exec(f"from z3 import And, Or\ndesired_output = {desired_output_str}", variables, local_dict)
    return local_dict['desired_output']


# Parse the file
variables_str, constraints_str, desired_output_str = parse_z3_file(z3_converted_file_path)

# Create Z3 variables
variables = create_variables(variables_str)
globals().update(variables)  # Make the variables available in the global namespace

# Create Z3 constraints
constraints = create_constraints(constraints_str, variables)

# Create Z3 desired output constraint
desired_output = create_desired_output(desired_output_str, variables)

# Create a Z3 solver instance
solver = z3.Solver()

# Add constraints to the solver
solver.add(constraints)
solver.add(desired_output)

# Check satisfiability
result = solver.check()

# Print result
if result == z3.sat:
    print("Satisfiable")
    model = solver.model()
    print("Model:", model)
elif result == z3.unsat:
    print("Unsatisfiable")
else:
    print("Unknown")
