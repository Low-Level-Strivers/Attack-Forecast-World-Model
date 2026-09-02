import numpy as np
import torch
import joblib
from src.preprocess import load_and_preprocess, create_sequences
from src.train_world_model import LSTMWorldModel

# 1. Load Monday data
state_df, feature_cols = load_and_preprocess("data/Monday-WorkingHours.pcap_ISCX.csv", nrows=50000)
scaler = joblib.load("scaler.pkl")
X_seq, y_next, _, _ = create_sequences(state_df, feature_cols, seq_len=10, scaler=scaler, is_train=False)

# 2. Compute MSE predictions
model = LSTMWorldModel(input_dim=len(feature_cols), hidden_dim=64)
model.load_state_dict(torch.load("world_model.pth"))
model.eval()

deviations = []
with torch.no_grad():
    for i in range(len(X_seq)):
        tensor_in = torch.tensor(X_seq[i], dtype=torch.float32).unsqueeze(0)
        pred = model(tensor_in).squeeze(0).numpy()
        mse = float(np.mean((y_next[i] - pred) ** 2))
        deviations.append(mse)

mean_mse = np.mean(deviations)
max_mse = np.max(deviations)

print(f"Monday Mean Loss (D_t): {mean_mse:.6f}")
print(f"Monday Max Loss  (D_t): {max_mse:.6f}")
print(f"Recommended Threshold : {mean_mse + (2 * np.std(deviations)):.6f}")