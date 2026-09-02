import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import joblib
from src.preprocess import load_and_preprocess

def train_baseline():
    print("Loading data for Logistic Regression Baseline...")
    
    # Load 1-minute state aggregation vectors from normal traffic (Monday) and attack traffic (Tuesday)
    df_mon, feature_cols = load_and_preprocess("data/Monday-WorkingHours.pcap_ISCX.csv", nrows=100000)
    df_tue, _ = load_and_preprocess("data/Tuesday-WorkingHours.pcap_ISCX.csv", nrows=100000)
    
    # Combine datasets for tabular benchmark training
    full_df = pd.concat([df_mon, df_tue], ignore_index=True)
    
    # Load scaler generated during preprocessing
    scaler = joblib.load("scaler.pkl")
    
    X = full_df[feature_cols].values
    y = full_df['is_attack'].values
    
    # Apply identical scaling transformation
    X_scaled = scaler.transform(X)
    
    # Train Logistic Regression model on individual 1-minute state vectors
    print(f"Training Logistic Regression on {len(full_df)} total minute state vectors...")
    clf = LogisticRegression(random_state=42)
    clf.fit(X_scaled, y)
    
    # Evaluate model predictions
    preds = clf.predict(X_scaled)
    
    print("\n--- Baseline Model Performance (Logistic Regression) ---")
    print(f"Precision : {precision_score(y, preds, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y, preds, zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(y, preds, zero_division=0):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y, preds, target_names=["Normal Traffic", "Attack Traffic"], zero_division=0))
    
    # Save trained baseline model
    joblib.dump(clf, "baseline_model.pkl")
    print("Baseline model saved to baseline_model.pkl")

if __name__ == "__main__":
    train_baseline()