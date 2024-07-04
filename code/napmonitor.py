from dd.autoref import BDD
import numpy as np 
from monitor import *
import pickle

class NAP_Monitor(BddMonitor):
        
    def __init__(self, numberOfClasses, numberOfNeurons, log, omittedNeuronIndices = {}):
    
        BddMonitor.__init__(self, numberOfClasses, numberOfNeurons, log, omittedNeuronIndices)
        
        # Create BDD variables
        for i in range(numberOfNeurons):
            self.bdd.add_var("x"+str(i))

    def addAllNeuronPatternsToClass(self, neuronValuesNp, predictedNp, labelsNp, classToKeep, log):
        
        if (not (type(neuronValuesNp) == np.ndarray)) or (not (type(predictedNp) == np.ndarray)) or (not (type(labelsNp) == np.ndarray)):
            raise TypeError('Input should be numpy array')
        
        mat = np.zeros(neuronValuesNp.shape)
        abs = np.greater(neuronValuesNp, mat)

        for exampleIndex in range(neuronValuesNp.shape[0]): 
            if classToKeep == -1:
                if(predictedNp[exampleIndex] == labelsNp[exampleIndex]):
                    self.addOnOffPatternToClass(abs[exampleIndex,:], predictedNp[exampleIndex], log)
            else:
                if(predictedNp[exampleIndex] == labelsNp[exampleIndex] and labelsNp[exampleIndex] == classToKeep):
                    self.addOnOffPatternToClass(abs[exampleIndex,:], predictedNp[exampleIndex], log)
        return True

   


    def addOnOffPatternToClass(self, neuronOnOffPattern, classIndex, log):
        
        # Basic data sanity check
        if not (len(neuronOnOffPattern) == self.numberOfNeurons):            
            raise IndexError('Neuron pattern size is not the same as the number of neurons being monitored')
        
        # Prepare BDD constraint
        constraint = ''
        for i in range(self.numberOfNeurons) :
            if neuronOnOffPattern[i] or ((classIndex in self.omittedNeuronIndices) and (i in self.omittedNeuronIndices[classIndex])):
                if i == 0: 
                    constraint = constraint + " x"+str(i)
                else: 
                    constraint = constraint + " & x"+str(i)
            else:
                if i == 0: 
                    constraint = constraint + " !x"+str(i)
                else: 
                    constraint = constraint + " & !x"+str(i)

        if(log == "true"):
            print("constraint", constraint)
        
        self.coveredPatterns[classIndex] = self.coveredPatterns[classIndex] | self.bdd.add_expr(constraint)
        #print("self.coveredPatterns[classIndex]", self.coveredPatterns[classIndex].to_expr())
        return True  
        


    
        
    def isPatternContained(self, neuronValuesNp, classIndex):
        
        if not (type(neuronValuesNp) == np.ndarray):
            raise TypeError('Input should be numpy array')
        
        zero = np.zeros(neuronValuesNp.shape)
        neuronOnOffPattern = np.greater(neuronValuesNp, zero)
        return self.isOnOffPatternContained(neuronOnOffPattern, classIndex)
        
    
        
    
    def isOnOffPatternContained(self, neuronOnOffPattern, classIndex):
        
        # Basic data sanity check
        if not (len(neuronOnOffPattern) == self.numberOfNeurons):            
            raise IndexError('Neuron pattern size is not the same as the number of neurons being monitored')
        
        # Prepare BDD constraint
        constraint = ''
        for i in range(self.numberOfNeurons) :
            if neuronOnOffPattern[i] or ((classIndex in self.omittedNeuronIndices) and (i in self.omittedNeuronIndices[classIndex])) :
                if i == 0: 
                    constraint = constraint + " x"+str(i)
                else: 
                    constraint = constraint + " & x"+str(i)
            else:
                if i == 0: 
                    constraint = constraint + " !x"+str(i)
                else: 
                    constraint = constraint + " & !x"+str(i)
        if (self.coveredPatterns[classIndex] & self.bdd.add_expr(constraint)) == self.bdd.false :            
            return False
        else:
            return True

 
       

    def saveToFile(self, fileName):
        """ Store the monitor based on the specified file name.
        """   
        if not (type(fileName) is string) : 
            raise TypeError("Please provide a fileName in string")
        
        self.bdd.dump(fileName) 
    
    def saveMonitor(self, monitor, filePath):
        with open(filePath, "wb") as output:
            pickle.dump(monitor, output, pickle.HIGHEST_PROTOCOL)