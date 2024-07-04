import argparse
import logging as log
import os
import time

from iris_runtime_monitor import *
from mnist_runtime_monitor import *

from napmonitor import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys



def createMonitor(modelName, nbClasses, nbNeurones, data, log):
    monitor = NAP_Monitor(nbClasses, nbNeurones, log)
    
    start_time = time.time()
    
    if modelName in ["IRIS_1", "IRIS_2", "IRIS_3"]:
        model, accuracy = train_iris_model(modelName, monitor, data, log)
    elif modelName == "MNIST":
        model, accuracy = train_mnist_model(modelName, monitor, data, log)
    else:
        raise ValueError(f"Unknown model name: {modelName}")
    
    end_time = time.time()

    execution_time = end_time - start_time
    
    print(monitor.bdd)
    print("Execution time: {:.2f} seconds".format(execution_time))
    
    return monitor, model 


def save(saveTo):
    print("Not yet implemented")




def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Runtime Verifier  *** Create the Monitor *** ")
    parser.add_argument("-m", "--model", type=str, help="Path to the model or model name")
    parser.add_argument("-c", "--classes", type=int,  help="Number of classes for the ANN Model")
    parser.add_argument("-n", "--neurones", type=int    ,help="Number of neurones for the l-1 layer")
    parser.add_argument("-d", "--data", type=str, help="Path to the training data")
    parser.add_argument("-s", "--savePath", type=str, help="Path to save the Monitor") 
    parser.add_argument("-l", "--log", type=str, help="Display Monitor creation log") 
    
           
    args = parser.parse_args()

    if not args.model:
        parser.error("Please provide the path to the Keras model file using -m or --model")
        

    monitor, model = createMonitor(args.model, args.classes, args.neurones, args.data, args.log)
    
    
    if args.savePath:
        monitor.saveMonitor(monitor, args.savePath+"\\"+args.model+"_Monitor.pkl")
        print("Monitor saved in: ",args.savePath+"\\"+args.model+"_Monitor.pkl")
        
        path = args.savePath+"\\"+args.model+"_Model.ckpt"
        torch.save(model.state_dict(), path)
        
        print("Trained model saved in: ",path)
        

if __name__ == "__main__":
    main()
