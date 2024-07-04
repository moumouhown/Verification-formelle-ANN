from dd.autoref import BDD
import numpy as np 

class BddMonitor():
    
    def __init__(self, numberOfClasses, numberOfNeurons, log , omittedNeuronIndices = {}):
    
        self.bdd = BDD()
            
        self.numberOfClasses = numberOfClasses
        self.numberOfNeurons = numberOfNeurons
        self.omittedNeuronIndices = omittedNeuronIndices 
        self.coveredPatterns = {}
        self.log = log
        
        # For each classified class, create a BDD to be monitored
        for i in range(numberOfClasses):
            self.coveredPatterns[i] = self.bdd.false

    