def generate_vnn_lib(property_statement, input_ranges, desired_output, constraints):
    """
    Generates VNN-LIB format for the given property statement, input ranges, desired output, and constraints.

    Args:
    property_statement (str): The property statement in Hoare logic.
    input_ranges (list): List of tuples containing variable names
    desired_output (str): The desired output statement.
    constraints (list): List of constraints.

    Returns:
    str: VNN-LIB format for the given inputs.
    """
    vnn_lib_format = ""

    # Declaration of variables
    vnn_lib_format += "; Declaration of variables of interest\n"
    declared_vars = set()  # To keep track of variable names already declared
    for var_declaration in input_ranges:
        var_name = var_declaration
        if var_name in declared_vars:
            print(f"Variable '{var_name}' has already been declared. Skipping...")
            continue
        vnn_lib_format += "(declare-fun " + var_name + "() Real)\n"
        declared_vars.add(var_name)
    vnn_lib_format += "\n"

    # Constraints
    vnn_lib_format += "; Constraints\n"
    for constraint in constraints:
        constraint_parts = constraint.split(' ')
        if len(constraint_parts) < 3:
            print(f"Invalid constraint format: {constraint}. Skipping...")
            continue
        try:
            parsed_constraint =  parse_constraint(constraint)
            formatted_constraint = f"(assert {parsed_constraint})\n"
        except Exception as e:
            print(f"Error parsing constraint '{constraint}': {str(e)}")
            continue  # Skip to the next constraint if parsing fails
        vnn_lib_format += formatted_constraint
    vnn_lib_format += "\n"

    # Desired output
    vnn_lib_format += "; Desired output\n"
    desired_output_parts = desired_output.split(' ')
    if len(desired_output_parts) != 3:
        print(f"Invalid desired output format: {desired_output}. Skipping...")
    else:
        var_name, op, value = desired_output_parts
        formatted_desired_output = f"(assert {op} {var_name} {value})\n"
        vnn_lib_format += formatted_desired_output

    return vnn_lib_format


def generate_vnn_lib_wrapper(property_file_path: str = None):
    """
    Prompts the user to define the property statement, input ranges, constraints, and desired output,
    then generates the corresponding VNN-LIB format.

    Args:
    property_file_path (str, optional): Path to the file containing the Hoare logic property. Defaults to None.
    """
    if property_file_path:
        try:
            with open(property_file_path, 'r') as file:
                property_statement = file.read().strip()
        except FileNotFoundError:
            print(f"File '{property_file_path}' not found.")
            return
        except Exception as e:
            print(f"Error reading file '{property_file_path}': {e}")
            return

        # Initialize variables
        input_ranges = []
        constraints = []
        desired_output = ""

        # Parse property_statement for input_ranges, constraints, and desired_output
        property_parts = property_statement.split(";")
        for part in property_parts:
            if "define:" in part.lower():
                # Parse input_ranges
                input_ranges.extend([x.strip() for x in part.split("define:")[1].split(",") if x.strip()])
            elif "constraint:" in part.lower():
                # Parse constraints
                constraints.extend([x.strip() for x in part.split("constraint:")[1].split(",") if x.strip()])
            elif "output:" in part.lower():
                # Parse desired_output
                desired_output = part.split("output:")[1].strip()

        # Generate VNN-LIB format
        vnn_lib_code = generate_vnn_lib(property_statement, input_ranges, desired_output, constraints)
        print("\nGenerated VNN-LIB format:\n")
        print(vnn_lib_code)
    else:
        # Proceed with the regular workflow for user input
        # Prompt user to define the property statement
        property_statement = input("Enter the property name in Hoare logic: ")

        # Prompt user to define variables
        print("\nDefine variables:")
        input_ranges = []
        while True:
            var_name = input("Enter the variable name (or type 'done' to finish): ")
            if var_name.lower().strip() == 'done':
                if len(input_ranges) > 0:
                    break
                else:
                    print("Please enter at least one variable.")
                    continue
            if any(var_name == var for var in input_ranges):
                print(f"Variable '{var_name}' has already been defined. Please enter a new variable name.")
                continue
            if(len(var_name.split(' ')) > 1 or is_math_type(var_name[0])):
                print(f"Variable '{var_name}' is not well defined. Please enter a new variable name.")
                continue
            input_ranges.append(var_name)

        # Prompt user to specify constraints
        print("\nDefine constraints:")
        constraints = []
        while True:
            constraint = input("Enter a constraint in space separated words (or type 'done' to finish): ")
            if constraint.lower().strip() == 'done':
                if len(constraints) > 0:
                    break
                else:
                    print("Please enter at least one constraint.")
                    continue
            if (not constraint_verif(constraint, input_ranges)):
                continue
            constraints.append(constraint)

        # Prompt user for desired output
        while True:
            desired_output = input("\nEnter the desired output: ")
            if constraint_verif(desired_output, input_ranges):
                break

        # Generate VNN-LIB format
        vnn_lib_code = generate_vnn_lib(property_statement, input_ranges, desired_output, constraints)
        print("\nGenerated VNN-LIB format:\n")
        print(vnn_lib_code)

def constraint_verif(statement, input_var):
    statement_verif = statement.split(' ')
    log_op = ['<', '>', '<=', '>=', '=', '!=', 'AND', 'OR', '(', ')']  # Ajout des parenthèses comme opérateurs logiques
    ari_op = ['+', '-', '*', '/']
    op = log_op + ari_op
    if len(statement_verif) < 3:
        print(f"Invalid statement '{statement}'. Please enter a valid statement.")
        return False
    elif not any((statement_word in log_op for statement_word in statement_verif) 
        or any((statement_word in op and 
                (statement_verif[statement_verif.index(statement_word) - 1] in op or statement_verif[statement_verif.index(statement_word) + 1] in op) 
            and (statement_verif.index(statement_word) > 0 and statement_verif.index(statement_word) < len(statement_verif)-1) ) 
        for statement_word in statement_verif) ):
        print(f"Invalid statement '{statement}'. Please enter a valid statement.")
        return False
    else:
        # Vérifiez si les variables de contrainte sont définies ou non
        for statement_word in statement_verif:
            if statement_word in op or statement_word in input_var or is_math_type(statement_word):
                continue
            else:
                print(f"variable not defined in '{statement}'. Please enter a valid statement.")
                return False
        return True
def is_math_type(word):
    # Check if it's a number
    if word.isdigit():
        return True

    # Check if it's a float
    try:
        float(word)
        return True
    except ValueError:
        pass

    # Add checks for other mathematical types as needed

    return False
import re

def parse_constraint(constraint):
    """
    Recursive function to parse a single constraint and convert it to the desired output format.

    Args:
        constraint: The constraint string to be parsed.

    Returns:
        A string representing the parsed constraint in the desired output format.
    """

    constraint = constraint.strip()
    print("Parsing constraint:", constraint)  # Debug print

    # Check if the constraint is a simple constraint
    constraint_list = constraint.split(' ')
    print("length : ",  len(constraint_list))
    if  len(constraint_list) == 3:
        var_name, op, value = constraint_list[0], constraint_list[1], constraint_list[2]
        return f"{op} {var_name} {value}"

    # Find the innermost set of parentheses
    start_index = constraint.rfind("(")
    end_index = constraint.find(")", start_index)

    while start_index != -1 and end_index != -1:
        # Parse the expression within the innermost set of parentheses
        inner_constraint = constraint[start_index + 1:end_index]
        inner_result = parse_constraint(inner_constraint)

        # Replace the expression within parentheses with the result
        constraint = constraint[:start_index] + inner_result + constraint[end_index + 1:]

        # Find the next innermost set of parentheses
        start_index = constraint.rfind("(")
        end_index = constraint.find(")", start_index)

    # Splitting by logical operators outside parentheses
    parts = constraint.split(" AND ") if " AND " in constraint else constraint.split(" OR ")

    if len(parts) > 1:
        logical_operator = "AND" if " AND " in constraint else "OR"
        processed_parts = [parse_constraint(part.strip()) for part in parts]
        return f"{logical_operator} {' '.join(processed_parts)}"
    else:
        # Simple constraint (handled by regex earlier)
        return constraint