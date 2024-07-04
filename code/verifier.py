import argparse
import logging as log
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
from loadModel import loadModel
from modelANN import NN  # Import the NN function from your module
from python_VNN_LIB_Wrapper import generate_vnn_lib_wrapper
from Z3_Wrapper import parse_input_file, generate_z3_script
import tensorflow as tf
from custom_loss import asymMSE  # Import your custom loss function
from SmtVerification import *
def load_model(model_path):
    # Load the model
    model_loader = loadModel(model_path)
    if model_loader.load_model():
        model = model_loader.get_model()  # Get the loaded Keras model
        
    else:
        log.error("Failed to load model!")
        sys.exit(1)


def load_model_and_print(model):
    # Load the model
    model_loader = loadModel(model)
    if model_loader.load_model():
        model = model_loader.get_model()  # Get the loaded Keras model
        # Print number of input neurons
        input_shape = model.input_shape[1:]  # Exclude batch size
        num_input_neurons = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        print("Number of input neurons:", num_input_neurons)

        # Print number of output neurons
        output_shape = model.output_shape[1:]  # Exclude batch size
        num_output_neurons = output_shape[0] if isinstance(output_shape, tuple) else output_shape
        print("Number of output neurons:", num_output_neurons)

        # Print number of layers
        num_layers = len(model.layers)
        print("Number of layers:", num_layers)
        print(model.summary())
        
        
        
        model_config = model.get_config()
        # Extraire les informations spécifiques
        model_name = model_config['name']
        trainable = model_config['trainable']
        layers_info = []
        print("~~~~~")      
        # Parcourir les couches
        for layer_config in model_config['layers']:
            layer_name = layer_config['class_name']
            layer_trainable = layer_config['config'].get('trainable', False)  # Utiliser get() avec une valeur par défaut
            
            # Initialiser un dictionnaire pour stocker les informations sur la couche
            layer_info = {'name': layer_name, 'trainable': layer_trainable}
            
            # Ajouter des informations spécifiques en fonction du type de couche
            if layer_name == 'InputLayer':
                input_shape = layer_config['config'].get('batch_shape')
                #print(layer_config['config'].get('batch_shape'))
                if input_shape:
                    num_neurons = input_shape[1] # Le deuxième élément de input_shape correspond au nombre de neurones
                    layer_info['num_neurons'] = num_neurons
                    layer_info['activation'] = layer_config['config'].get('activation', 'None')
                else:
                    layer_info['num_neurons'] = None
                    layer_info['activation'] = None
            elif layer_name == 'Dense':
                num_neurons = layer_config['config'].get('units')
                layer_info['num_neurons'] = num_neurons
                layer_info['activation'] = layer_config['config'].get('activation', 'None')
            elif layer_name == 'OutputLayer':
                num_neurons = layer_config['config'].get('units')
                layer_info['num_neurons'] = num_neurons
                layer_info['activation'] = layer_config['config'].get('activation', 'None')
            
            # Ajouter les informations de la couche à la liste des couches
            layers_info.append(layer_info)

        # Imprimer les informations
        print("Nom du modele:", model_name)
        print("Modele entrainable:", trainable)
        print("Informations sur les couches:")
        # Parcourir les informations sur les couches et les imprimer
        for layer_info in layers_info:
            print("  - Nom de la couche:", layer_info['name'])
            print("    Entrainable:", layer_info['trainable'])
            
            # Imprimer des informations supplémentaires spécifiques à chaque couche
            if layer_info['name'] == 'InputLayer':
                print("    Nombre de neurones:", layer_info.get('num_neurons', 'N/A'))
                print("    Fonction d'activation:", layer_info.get('activation', 'N/A'))
            elif layer_info['name'] == 'Dense' or layer_info['name'] == 'OutputLayer':
                print("    Nombre de neurones:", layer_info.get('num_neurons', 'N/A'))
                print("    Fonction d'activation:", layer_info.get('activation', 'N/A'))  
                
    else:
        log.error("Failed to load model!")
        sys.exit(1)

def convert_to_constraints(model):
    # Load the model
    model_loader = loadModel(model)
    if model_loader.load_model():
        model = model_loader.get_model()  # Get the loaded Keras model
        inputs, outputs = NN(model_loader.get_model())
        # Print inputs
        #print("================================================")
        print("Inputs:", inputs)
        print("Outputs:", outputs)
      
    else:
        log.error("Failed to load model!")
        sys.exit(1)

def smt_verification(model,propertyPath):
    model_loader = loadModel(model)
    if model_loader.load_model():
        inputs, outputs = NN(model_loader.get_model())
        verification(inputs,outputs,propertyPath)
    else:
        log.error("Failed to load model!")
        sys.exit(1)
    

def smt_verification_robustness(model,propertyPath, epsVal):
    model_loader = loadModel(model)
    if model_loader.load_model():
        inputs, outputs = NN(model_loader.get_model())
        verification_robustness(inputs,outputs,propertyPath)
    else:
        log.error("Failed to load model!")
        sys.exit(1)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Model Verifier")
    parser.add_argument("-m", "--model", type=str, help="Path to the Keras model file")
    parser.add_argument("-p", "--print-parameters", action="store_true", help="Print model parameters")
    parser.add_argument("-c", "--convert-to-constraints", action="store_true", help="Convert model to constraints")
    parser.add_argument("-v", "--convert-property-to-vnnLib-format", action="store_true", help="convert property to vnn-Lib format")
    parser.add_argument("-f", "--convert-property-file-to-vnnLib-format", type=str, help="convert property file to vnn-Lib format")   
    parser.add_argument("-s", "--smt-verification-property", type=str, help="verify ann using smt technique")       
    parser.add_argument("-r", "--eps", type=str, help="Epsilon value for robustness property check")       
    
    args = parser.parse_args()

    #if not args.model:
        #parser.error("Please provide the path to the Keras model file using -m or --model")

    # Load the model only if any action is requested
    if args.model:
        model = load_model(args.model)
      
    if args.print_parameters:
        # Print model parameters
        load_model_and_print(args.model)

    if args.convert_to_constraints:
        convert_to_constraints(args.model)

    if args.convert_property_to_vnnLib_format and args.convert_property_file_to_vnnLib_format:
        generate_vnn_lib_wrapper(args.convert_property_file_to_vnnLib_format)
        preconditions, domain_constraints, postconditions= parse_input_file(args.convert_property_file_to_vnnLib_format)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        generate_z3_script(preconditions, domain_constraints, postconditions)
    elif args.convert_property_to_vnnLib_format:
        generate_vnn_lib_wrapper()
        
    if args.smt_verification_property and args.eps:
        smt_verification_robustness(args.model, args.smt_verification_property, args.eps)
        
    elif args.smt_verification_property:
        smt_verification(args.model, args.smt_verification_property)
    

if __name__ == "__main__":
    main()
