import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
CSV_PATH = "data/data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.bin")
WINDOW_SIZE = 60
HIDDEN_SIZE = 32
EPOCHS = 6000 # You can reduce this for testing
LEARNING_RATE = 0.001
PRINT_EVERY = 10000

# --- Model Definition ---
class TradingModel(nn.Module):
    def __init__(self):
        super(TradingModel, self).__init__()
        self.fc1 = nn.Linear(WINDOW_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, 1)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def main():
    print("[INFO] Starting Training Pipeline...")

    # 1. Load Data
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] {CSV_PATH} not found. Run get_data.py first.")
        return

    data = pd.read_csv(CSV_PATH)
    closes = data[["Close"]].values
    print(f"[INFO] Loaded {len(closes)} rows from CSV.")

    # 2. Scale Data (Normalization 0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closes)

    # 3. Create Sliding Windows
    X_train = []
    y_train = []

    for i in range(WINDOW_SIZE, len(scaled_data)):
        X_train.append(scaled_data[i-WINDOW_SIZE : i])
        y_train.append(scaled_data[i, 0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Convert to PyTorch Tensors
    x_torch = torch.tensor(X_train).float().view(-1, WINDOW_SIZE)
    y_torch = torch.tensor(y_train).float().view(-1, 1)

    print(f"[INFO] Training Data Shape: X={x_torch.shape}, y={y_torch.shape}")

    # 4. Initialize Model
    model = TradingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE , weight_decay=1e-5)

    # 5. Training Loop
    print(f"[TRAIN] Starting training for {EPOCHS} epochs...")
    
    for i in range(EPOCHS):
        outputs = model(x_torch)
        loss = criterion(outputs, y_torch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % PRINT_EVERY == 0:
            print(f'[TRAIN] Epoch [{i+1}/{EPOCHS}], Loss: {loss.item():.6f}')

    print("[TRAIN] Training completed.")

    # 6. Export to Binary (for C)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"[EXPORT] Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        # A. Save Scaler Config (Min/Max) first
        min_val = scaler.data_min_[0]
        max_val = scaler.data_max_[0]
        np.array([min_val, max_val], dtype=np.float32).tofile(f)
        print(f"[EXPORT] Scaler config saved: Min={min_val:.4f}, Max={max_val:.4f}")

        # B. Save Weights and Biases
        for param in model.parameters():
            data_numpy = param.detach().numpy()
            data_numpy.tofile(f)
            print(f"[EXPORT] Layer saved. Shape: {data_numpy.shape}")

    print("[SUCCESS] Pipeline finished.")

if __name__ == "__main__":
    main()