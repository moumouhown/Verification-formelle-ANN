from constraint_verifier.constraint_verifier import constraint_verifier
valid_exp=[]
def remove_additional_spaces(s):
    # Split the string into words
    words = s.split()
    # Join the words with a single space
    cleaned_string = ' '.join(words)
    return cleaned_string
import re
def add_space_around_operators(s):
    # Define logical and arithmetic operators
    log_op = ['<', '>', '<=', '>=', '==', '!=', 'AND', 'OR', '(', ')']
    ari_op = ['+', '-', '*', '/']

    # Create regex patterns for logical and arithmetic operators
    for op in log_op + ari_op:
        # Escape operators that are special characters in regex
        op_escaped = re.escape(op)
        s = re.sub(rf'\s*{op_escaped}\s*', f' {op} ', s)
    
    # Remove any extra spaces
    s = ' '.join(s.split())
    return s
def get_verified_constraints(constraints):
    verified_constraints=[]
    for i in range(len(constraints)):
        constraints[i] = constraints[i] .replace("AND", "And")
        constraints[i] = constraints[i] .replace("OR", "Or")
        constraints[i] = remove_additional_spaces(constraints[i])
        #print("constraint verifier : "+str(constraint_verifier(constraints[i])))
        if (not constraint_verifier(constraints[i])):
            continue
        verified_constraints.append(constraints[i])
    return verified_constraints
def parse_input_file(input_filename):
    with open(input_filename, 'r') as file:
        lines = file.readlines()
        for line in lines :
            if line.find("precondition:") != -1:
                precondition_line = line.strip()
                # Extracting preconditions
                preconditions = precondition_line.split(':')[1].strip().split(', ')
                preconditions[-1] = preconditions[-1][:-1]
            elif line.find("domain_constraints:") != -1:
                domain_constraints_line = line.strip()
                # Extracting domain_constraints
                domain_constraints = domain_constraints_line.split(':')[1].strip().split(', ')
                domain_constraints[-1] = domain_constraints[-1][:-1]
            elif line.find("postcondition:") != -1:
                postcondition_line = line.strip()
                # Extracting postconditions
                postconditions = postcondition_line.split(':')[1].strip().split(', ')
                postconditions[-1] = postconditions[-1][:-1]

    preconditions= get_verified_constraints(preconditions)
    domain_constraints= get_verified_constraints(domain_constraints)
    postconditions = get_verified_constraints(postconditions)


    return preconditions, domain_constraints, postconditions
def get_var_name(var):
    #print(globals().items())
    for name, value in globals().items():
        if value is var:
            return name
    return None
def write_z3_constraints_in_file(constraints, file, constraint_name):
    # Writing the constraints
    constraint_var_name = get_var_name(constraints)
    if constraint_var_name == None:
        constraint_var_name = constraint_name
    #file.write(f"# {constraint_var_name}\n")
    for i in range(len(constraints)):
        #constraints[i]= add_space_around_operators(constraint)
        constraints[i]=parse_constraint(constraints[i])
        constraints[i] = constraints[i].replace("_", "")
        constraints[i]= remove_additional_spaces(constraints[i])
        
    if constraint_name != "postconditions" :
        constraints_str = "And(" + ", ".join(constraints) + ")"
    else:
        constraints_str = "And(Not(" + ", ".join(constraints) + "))"

    """
    if constraint_name == "preconditions" :
        constraints_str = "And(" + ", ".join(constraints) + ")"
    elif constraint_name == "domain_constraints" :
        constraints_str =  ", ".join(constraints)
    else:
        constraints_str = ", And(Not(" + ", ".join(constraints) + ")))"
    """
    
    return constraints_str
    
    
def generate_z3_script(preconditions, domain_constraints, postconditions, output_filename='z3_script.py'):
    with open(output_filename, 'w') as file:
        # Writing the header
        #file.write("from z3 import *\n\n")
        pre = write_z3_constraints_in_file(preconditions, file, "preconditions")
        dom = write_z3_constraints_in_file(domain_constraints, file, "domain_constraints")
        post = write_z3_constraints_in_file(postconditions, file, "postconditions")
        
        print("And("+pre +","+ post +","+ dom+")")
        
def find_and_or_operator_in_list(constraint_list):
    for i, token in enumerate(constraint_list):
        if token in ['and', 'or']:
            return i
    return -1

def find_last_opening_parenthesis(constraint_list):
    for i in range(len(constraint_list) - 1, -1, -1):
        if '(' in constraint_list[i]:
            return i
    return -1

def parse_constraint(constraint):
    constraint_list = constraint.split()
    #print("parsing constraint:", constraint_list)
    and_or_index = find_and_or_operator_in_list(constraint_list)
    
    if and_or_index != -1:
        last_op_par_index = find_last_opening_parenthesis(constraint_list[:and_or_index])
        
        if last_op_par_index == -1:
            last_op_par_index = 0
            constraint_list[0] =  constraint_list[0]
            constraint_list[-1] = constraint_list[-1] 
        
        if not(constraint_list[last_op_par_index].find("(") == 0):
            # Capitalize the operator and move it to the appropriate position
            constraint_list[last_op_par_index] = constraint_list[and_or_index].capitalize() + "(" + constraint_list[last_op_par_index]
            constraint_list[and_or_index] = ","
            constraint_list.append(")")
        else:
            # Capitalize the operator and move it to the appropriate position
            constraint_list[last_op_par_index] = constraint_list[and_or_index].capitalize() +  constraint_list[last_op_par_index]
            constraint_list[and_or_index] = ","
        
        #print("the result of parsing:", " ".join(constraint_list))
        return parse_constraint(" ".join(constraint_list))
    else:
        return constraint

if __name__ == "__main__":
    # Example usage
    input_filename = 'testing.txt'  # Replace with your input file name
    preconditions, domain_constraints, postconditions= parse_input_file(input_filename)
    generate_z3_script(preconditions, domain_constraints, postconditions)