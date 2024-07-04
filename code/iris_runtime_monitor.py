import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

def get_iris_dataset(datasetPath):
    df = pd.read_csv(datasetPath)
    
    # Extract features and target
    x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
    y = df['Species'].astype('category').cat.codes.values  # Convert species to numeric codes
       
    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x, dtype=torch.float32)
    y_train_tensor = torch.tensor(y, dtype=torch.long)
    
    return x_train_tensor, y_train_tensor

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





def train_iris_model(modelName, monitor, datasetPath, log):
    x_train, y_train= get_iris_dataset(datasetPath)
    
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    if modelName == "IRIS_1":
        model = IrisNet1()
    elif modelName == "IRIS_2":
        model = IrisNet2()
    elif modelName == "IRIS_3":
        model = IrisNet3()
        
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training the model
    epochs = 1
    
    for epoch in range(epochs):
        correct = 0
        total = 0
        if(log == "true"):
            print("\n\nepoch",epoch,"/",epochs)
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs, intermediate_values = model.forward_with_intermediate(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            
            if(log == "true"):
                print("y_batch",y_batch)
            
            if(predicted == y_batch).sum().item():
                if(log == "true"):
                    print("x_batch",x_batch)
                    print("predicted ",predicted)
                    print("intermidiate values ",intermediate_values)
                
                monitor.addAllNeuronPatternsToClass(intermediate_values.detach().numpy(), 
                                                predicted.detach().numpy(), 
                                                y_batch.detach().numpy(), 
                                                -1, 
                                                log)
                
    accuracy = correct / total
    
    print("\nModel accuracy: ", accuracy)
    return model, accuracy