# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
import numpy as np

class LSTMWorldModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMWorldModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train():
    X_train = np.load("X_seq_train.npy")
    y_train = np.load("y_next_train.npy")
    
    input_dim = X_train.shape[2]
    model = LSTMWorldModel(input_dim=input_dim, hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    print("Training World Model on 1-Minute State Sequences...")
    epochs = 80
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Training MSE Loss: {loss.item():.6f}")
            
    torch.save(model.state_dict(), "world_model.pth")
    print("Saved trained weights to world_model.pth")

if __name__ == "__main__":
    train()