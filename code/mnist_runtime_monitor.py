import numpy.random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy.misc
import math
import matplotlib.pyplot as plt
from torch.autograd  import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.inception import inception_v3
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F



def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)        


def data_load(pathData):
    # MNIST dataset 
    batch_size = 1
    train_dataset = torchvision.datasets.MNIST(root=pathData, 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=True)

    test_dataset = torchvision.datasets.MNIST(root=pathData, 
                                            train=False, 
                                            transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)
    
    return train_loader 



class NeuralNet(nn.Module):
    
    
    
    def __init__(self):
        super(NeuralNet, self).__init__()
        num_classes = 10
        sizeOfNeuronsToMonitor = 40
 
        self.conv1 = nn.Conv2d(1, 40, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(40, 20, 5)
        self.fc1 = nn.Linear(20 * 4 * 4, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80, sizeOfNeuronsToMonitor)
        self.fc4 = nn.Linear(sizeOfNeuronsToMonitor, num_classes)
        
    def forward(self, x):
        # Original 28x28x1 -(conv)-> 24x24x40 -(pool)-> 12x12x40
        x = self.pool(F.relu(self.conv1(x)))
        # Original 12x12x40 -(conv)-> 8x8x20 -(pool)-> 4x4x20
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten it to an array of inputs
        x = x.view(-1, 20 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc4(x)
        return out 
  
    # Here we add another function, which does the same forward computation but also extracts intermediate layer results
    def forwardWithIntermediate(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        intermediateValues = x
        x = F.relu(x)
        out = self.fc4(x)
        return out, intermediateValues    
    



def train_mnist_model(modelName, monitor, datasetPath, log): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10
    num_epochs = 5
    batch_size = 40
    learning_rate = 0.001
    sizeOfNeuronsToMonitor = 40
    
    net = NeuralNet()
    model_path = os.path.join(os.path.dirname(__file__), 'models/1_model_MNIST_CNN.ckpt')
    net.load_state_dict(torch.load(model_path))
    net.eval()
    
    
    
    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) 
    train_loader = data_load(datasetPath)
    
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in train_loader:
            labels = labels.to(device)
            outputs, intermediateValues = net.forwardWithIntermediate(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            monitor.addAllNeuronPatternsToClass(intermediateValues.numpy(), predicted.numpy(), labels.numpy(), -1, log)

        accuracy = 100 * correct / total
                    
        print('\nModel accuracy: {} %'.format(100 * correct / total))
    return net, accuracy



