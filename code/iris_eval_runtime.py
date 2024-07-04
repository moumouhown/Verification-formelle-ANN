import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import argparse
import logging as log
import os
import time
import pickle 
from napmonitor import *

def get_iris_dataset(datasetPath):
    df = pd.read_csv(datasetPath)
    
    # Extract features and target
    x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
    y = df['Species'].astype('category').cat.codes.values  # Convert species to numeric codes
       
    # Convert to PyTorch tensors
    x_test_tensor = torch.tensor(x, dtype=torch.float32)
    y_test_tensor = torch.tensor(y, dtype=torch.long)
    
    return x_test_tensor, y_test_tensor

class IrisNet1(nn.Module):
    def __init__(self):
        super(IrisNet1, self).__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def forward_with_intermediate(self, x):
        x = torch.relu(self.fc1(x))
        intermediate_values = x.detach()  # Detach here
        out = self.fc2(x)
        return out, intermediate_values



class IrisNet2(nn.Module):
    def __init__(self):
        super(IrisNet2, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def forward_with_intermediate(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        intermediate_values = x.detach()  # Detach here
        out = self.fc4(x)
        return out, intermediate_values


class IrisNet3(nn.Module):
    def __init__(self):
        super(IrisNet3, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def forward_with_intermediate(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        intermediate_values = x.detach()  # Detach here
        out = self.fc4(x)
        return out, intermediate_values


def evaluate_iris_model(modelName, modelPath, monitor, datasetPath):
    x_test, y_test= get_iris_dataset(datasetPath)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    if modelName == "IRIS_1":
        model = IrisNet1()
        
    elif modelName == "IRIS_2":
        model = IrisNet2()
    elif modelName == "IRIS_3":
        model = IrisNet3()
    
    model.load_state_dict(torch.load(modelPath))

    with torch.no_grad():
        correct = 0
        outofActivationPattern = 0
        outofActivationPatternAndResultWrong = 0
       
        total = 0
        
        for x_batch, y_batch in test_loader:
            outputs, intermediateValues = model.forward_with_intermediate(x_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
                   
            # Additional processing for runtime monitoring
            predictedNp = predicted.numpy()
            result = (predicted == y_batch)
            res = result.numpy()
                   
            for exampleIndex in range(intermediateValues.shape[0]):  
                if not monitor.isPatternContained(intermediateValues.numpy()[exampleIndex,:], predicted.numpy()[exampleIndex]):
                    outofActivationPattern = outofActivationPattern +1
                    if res[exampleIndex] == False :
                        outofActivationPatternAndResultWrong = outofActivationPatternAndResultWrong + 1

        print('Total number of Operational Data: ', total)
        
        print('Number of Operational Data correctly predicted by the model : ', correct)
        
        print('Number of Operational Data wrong predicted by the model : ', total - correct)
        
        print('Accuracy of the Model on Operational Data: {} %'.format(100 * correct / total))
        
        print('Number of Operatioanl Data Decided as incorrect by the monitor: ', total - outofActivationPattern + outofActivationPatternAndResultWrong)
        
        #print('outofActivationPatternAndResultWrong: ', outofActivationPatternAndResultWrong)
        
        #print('Number of Operatioanl Data out of the Monitor and wrong predicted by the Model: ', outofActivationPattern)
        
        #print('Out-of-activation pattern & misclassified / out-of-activation pattern : {} %'.format(100 * outofActivationPatternAndResultWrong / (outofActivationPattern)))
    
