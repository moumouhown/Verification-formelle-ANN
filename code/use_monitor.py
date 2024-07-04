import argparse
import logging as log
import os
import time
import pickle 

from iris_eval_runtime import *
from mnist_eval_runtime import *
from napmonitor import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

def get_directory_path(file_path):
    # Get the directory name from the file path
    directory_path = os.path.dirname(file_path)
    return directory_path



def get_Monitor(monitor_path):
    with open(monitor_path, 'rb') as input:
        monitor = pickle.load(input)
    
    return monitor



def dispalyBDD(monitorPath):
    monitor = get_Monitor(monitorPath)
    nb = len(monitor.coveredPatterns)
    for i in range(nb):
        path = get_directory_path(monitorPath)+'\BDD_'+str(i)+'.png' 
        print('-'+path)
        monitor.coveredPatterns[i].bdd.dump(path)
            
            

    

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Runtime Verifier  *** Create the Monitor *** ")
    parser.add_argument("-m", "--monitor", type=str, help="Path to the monitor")
    parser.add_argument("-d", "--dispaly", type=str, help="Dispaly BDDs of the monitor")  
    parser.add_argument("-n", "--model", type=str, help="Name of the model")
    parser.add_argument("-p", "--path", type=str, help="Path to the model")  
    parser.add_argument("-t", "--data", type=str, help="Path to the operational data")  
      
    
    
           
    args = parser.parse_args()

    if not args.monitor:
        parser.error("Please provide the path to the monitor")
    
    if args.dispaly:
        dispalyBDD(args.monitor)
    
    if args.model: 
        monitor = get_Monitor(args.monitor)
        modelName = args.model
        modelPath = args.path
        datasetPath = args.data
        if modelName in ["IRIS_1", "IRIS_2", "IRIS_3"]:
            evaluate_iris_model(modelName, modelPath, monitor, datasetPath)
        elif modelName == "MNIST":
            evaluate_mnist_model(modelName, modelPath, monitor, datasetPath)                                               
        else:
            raise ValueError(f"Unknown model name: {modelName}")
        
    
    
if __name__ == "__main__":
    main()
