And(And((inputs[0] >= 1 , inputs[1] >= 1) ))
And(And(Or((inputs[0] == 0 , inputs[0] == 1) , And(Or((inputs[1] == 0 , inputs[1] == 1) , Or((outputs[0] == 0 , outputs[0] == 1) ) ) ) ) ))
And(Not(outputs[0] == 1))
