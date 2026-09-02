import torch
import numpy as np
from src.train_world_model import LSTMWorldModel

class InferenceEngine:
    # Set to your measured calibration threshold (~0.096)
    def __init__(self, feature_dim=15, baseline_threshold=0.096):
        self.model = LSTMWorldModel(input_dim=feature_dim, hidden_dim=64)
        self.model.load_state_dict(torch.load("world_model.pth"))
        self.model.eval()
        self.threshold = baseline_threshold

    def _calc_prob(self, deviation):
        # Sigmoid center dynamic scaling tuned to baseline variance
        return float(1.0 / (1.0 + np.exp(-15 * (deviation - self.threshold))))

    def evaluate_sequence(self, input_seq, actual_future_state):
        curr_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred_state = self.model(curr_tensor).squeeze(0).numpy()
            
        # 1. State Deviation D_t
        dev_t = float(np.mean((actual_future_state - pred_state) ** 2))
        prob_t = self._calc_prob(dev_t)
        
        # 2. Multi-Step Autoregressive Prediction Horizon (S_{t+1}, S_{t+2}, S_{t+3})
        horizon_probs = [prob_t]
        rolling_seq = input_seq.copy()
        last_pred = pred_state
        
        for step in range(1, 4):
            rolling_seq = np.vstack([rolling_seq[1:], last_pred])
            seq_tensor = torch.tensor(rolling_seq, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                next_pred = self.model(seq_tensor).squeeze(0).numpy()
                
            step_dev = float(np.mean((next_pred - last_pred) ** 2)) + dev_t * (1 + 0.1 * step)
            horizon_probs.append(min(1.0, self._calc_prob(step_dev)))
            last_pred = next_pred

        # Threat Stage Mapping
        threat_level = "LOW"
        mitre_stage = "Normal Traffic"
        
        if prob_t > 0.7:
            threat_level = "HIGH"
        elif prob_t > 0.4:
            threat_level = "MEDIUM"

        if prob_t > 0.4:
            if actual_future_state[1] > 0.4:
                mitre_stage = "Reconnaissance"
            elif actual_future_state[8] > 0.4:
                mitre_stage = "Execution"
            elif actual_future_state[5] > 0.4:
                mitre_stage = "Lateral Movement"
            else:
                mitre_stage = "Initial Access"
                
        return dev_t, prob_t, horizon_probs, threat_level, mitre_stage