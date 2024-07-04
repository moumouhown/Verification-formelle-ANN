from constraint_verifier.constraint_verifier import constraint_verifier
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

    # Constraints
    vnn_lib_format += "; Constraints\n"
    for constraint in constraints:
        constraint = remove_additional_spaces(constraint)
        constraint_parts = constraint.split(' ')
        if len(constraint_parts) < 3:
            print(f"Invalid constraint format: {constraint}. Skipping...")
            continue
        try:
            parsed_constraint =  parse_constraint(constraint)
            parsed_constraint = remove_additional_spaces(parsed_constraint)
            parsed_constraint = parsed_constraint.replace("/", ")")
            parsed_constraint = parsed_constraint.replace("\\", "(")
            formatted_constraint = f"(assert ({parsed_constraint}) )\n"
        except Exception as e:
            print(f"Error parsing constraint '{constraint}': {str(e)}")
            continue  # Skip to the next constraint if parsing fails
        vnn_lib_format += formatted_constraint
    vnn_lib_format += "\n"


    return vnn_lib_format

def remove_additional_spaces(s):
    # Split the string into words
    words = s.split()
    # Join the words with a single space
    cleaned_string = ' '.join(words)
    return cleaned_string

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
            part= remove_additional_spaces(part)
            if ":" in part.lower():
                # Parse constraints
                constraints.extend([x.strip() for x in part.split(":")[1].split(",") if x.strip()])
                constraints_verified = []
                if len(constraints) > 0:
                    for constraint in constraints:
                        #constraint = add_space_around_operators(constraint)
                        if (not constraint_verifier(constraint)):
                            continue
                        constraints_verified.append(constraint)
                    constraints = constraints_verified
                else:
                    print("Please enter at least one constraint.")
                    continue

        # Generate VNN-LIB format
        vnn_lib_code = generate_vnn_lib(property_statement, input_ranges, desired_output, constraints)
        print("\nGenerated VNN-LIB format:\n")
        print(vnn_lib_code)
        # Write the generated VNN-LIB format to a file named "test.txt"
        with open("test.txt", "w") as file:
            file.write(vnn_lib_code)
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
            if (not constraint_verifier(constraint, input_ranges)):
                continue
            constraints.append(constraint)

        # Prompt user for desired output
        while True:
            desired_output = input("\nEnter the desired output: ")
            if constraint_verifier(desired_output, input_ranges):
                break

        # Generate VNN-LIB format
        vnn_lib_code = generate_vnn_lib(property_statement, input_ranges, desired_output, constraints)
        print("\nGenerated VNN-LIB format:\n")
        print(vnn_lib_code)

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
def add_space_around_operators(s):
    # Define logical and arithmetic operators
    log_op = ['<', '>', '<=', '>=', '=', '!=', 'AND', 'OR', '(', ')']
    ari_op = ['+', '-', '*', '/']

    # Create regex patterns for logical and arithmetic operators
    for op in log_op + ari_op:
        # Escape operators that are special characters in regex
        op_escaped = re.escape(op)
        s = re.sub(rf'\s*{op_escaped}\s*', f' {op} ', s)
    
    # Remove any extra spaces
    s = ' '.join(s.split())
    return s
def parse_constraint(constraint, operators = r'\b(and|or)\b'):
    """
    Recursive function to parse a single constraint and convert it to the desired output format.

    Args:
        constraint: The constraint string to be parsed.

    Returns:
        A string representing the parsed constraint in the desired output format.
    """

    constraint = constraint.strip()
    #print("Parsing constraint:", constraint)  # Debug print
    #print("operators : ", operators)
    # Check if the constraint is a simple constraint
    constraint_list = constraint.split(' ')
    if len(constraint_list) == 3:
        var_name, op, value = constraint_list[0], constraint_list[1], constraint_list[2]
        #print("sending constraint : ", f"{op} {var_name} {value}")
        return f"{op} {var_name} {value}".strip()

    # Find the innermost set of parentheses
    start_index = constraint.rfind("(")
    end_index = constraint.find(")", start_index)

    while start_index != -1 and end_index != -1:
        # Parse the expression within the innermost set of parentheses
        inner_constraint = constraint[start_index + 1:end_index]
        inner_result = "\\"+parse_constraint(inner_constraint)+"/"

        # Replace the expression within parentheses with the result
        constraint = constraint[:start_index] + inner_result + constraint[end_index + 1:]

        # Find the next innermost set of parentheses
        start_index = constraint.rfind("(")
        end_index = constraint.find(")", start_index)

    # Splitting by logical operators outside parentheses
    parts = re.split(operators, constraint)
    #print("parts : ", parts)
    if(len(parts) > 3):
        parts = adjust_parts(parts)
        #print("parts after first adjust : ", parts)
        if(len(parts) > 3):
            adjust_parts(parts)
            #print("parts after second adjust : ", parts)


    if len(parts) > 1 and len(parts[1].strip()) <= 3:
        logical_operator = parts[1].strip()
        #print("logical operator : "+logical_operator)
        processed_parts = [parse_constraint(part.strip()) for part in parts[::2]]  # parts[::2] selects operands
        #print("result : ", logical_operator, ' '.join(processed_parts))
        return f"{logical_operator} {' '.join(processed_parts)}".strip()
    else:
        # Simple constraint (handled by regex earlier)
        #if(len(parts[0]) > 1):
        #    return parse_constraint(constraint, r'[+\-*/]')
        #print("sending constraint : ", constraint)
        constraints = constraint.split(" ")
        #print("constraint : "+str(constraints))
        return constraint.strip()
def adjust_parts(parts):
    i = 1
    while i < len(parts):
        if len(parts[i].split(" "))> 3:
            parts[i-1] += parts[i]
            parts.pop(i)
        else:
            i += 1
    return parts
if __name__ == "__main__":
    generate_vnn_lib_wrapper("testing.txt")