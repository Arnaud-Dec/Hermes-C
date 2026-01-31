import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import os

csv_path = "data/data.csv"
model_path = "models/model.bin"

window_size = 60 #nb day

os.makedirs("models", exist_ok=True)

X_train =[]
y_train = []

# Clean
print(f"load {csv_path}...")
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    print(csv_path+ " Not Found")

closes = data[["Close"]].values

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closes)

# Sliding Window

for i in range(60 , len(scaled_data)):
    X_train.append(scaled_data[i-60 : i])
    y_train.append(scaled_data[i,0])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_torch = torch.tensor(X_train).float()
y_torch = torch.tensor(y_train).float()

# 2. On applique le view ET on écrase l'ancienne variable avec le résultat
X_torch = X_torch.view(-1, 60)
y_torch = y_torch.view(-1, 1)

# Vérification 
print("X shape:", X_torch.shape) 
print("y shape:", y_torch.shape) 

class TradingModel(nn.Module):
    def __init__(self):
        super(TradingModel, self).__init__()
        self.fc1 = nn.Linear(60,32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32,1)
        
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return (out)
    
model = TradingModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters() , lr=0.001)

print("--- Start Training ---")

for i in range(600000):
    outputs = model(X_torch)
    loss = criterion(outputs , y_torch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (i+1) % 10000 == 0:
        print(f'Epoch [{i+1}/600000], Loss: {loss.item():.6f}')

print("--- End Training ---")


print("\n--- Save Bin for C ---")

with open(model_path, "wb") as f:
    for param in model.parameters():
        data = param.detach() 
        data = data.numpy()
        data.tofile(f)
    
        print(f"Write: {data.shape}")
