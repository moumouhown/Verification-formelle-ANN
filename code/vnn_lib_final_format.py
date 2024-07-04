import re

def transform_to_prefix(expression):
    # Remove 'assert' and trim whitespace
    expression = expression.replace('assert', '').strip()

    def process_expression(expr):
        # Base case: if it's a simple comparison, return it directly
        if is_simple_comparison(expr):
            return expr
        
        # Extract the main logical operator and its operands
        operator, operands = extract_operator_and_operands(expr)
        
        # Recursively process each operand
        processed_operands = [process_expression(op) for op in operands]
        
        # Construct the prefix notation for the current expression
        return f'({operator} {" ".join(processed_operands)})'

    # Process the main expression
    return f'(assert {process_expression(expression)})'

def is_simple_comparison(expr):
    # Check if the expression is a simple comparison like '> x 0' or '< y 0'
    simple_comparison_pattern = r'^[<>]=? \w+ \d+$'
    return re.match(simple_comparison_pattern, expr.strip()) is not None

def extract_operator_and_operands(expr):
    # Find the main logical operator (AND, OR) and its operands
    # Split by AND and OR not within parentheses
    split_expr = re.split(r'\s+(AND|OR)\s+', expr, maxsplit=1)
    
    if len(split_expr) == 3:
        left_operand, operator, right_operand = split_expr
        # Handle nested expressions
        left_operands = split_operands(left_operand.strip())
        right_operands = split_operands(right_operand.strip())
        return operator, left_operands + right_operands
    else:
        return None, [expr.strip()]

def split_operands(expr):
    # Split operands that are within parentheses
    result = []
    balance = 0
    current_operand = []
    for char in expr:
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        if balance == 0 and char.isspace():
            result.append(''.join(current_operand).strip())
            current_operand = []
        else:
            current_operand.append(char)
    if current_operand:
        result.append(''.join(current_operand).strip())
    return result

# Example usage
expression = "(assert AND > x 0 OR < y 0 > x 10)"
transformed_expression = transform_to_prefix(expression)
print(transformed_expression)
