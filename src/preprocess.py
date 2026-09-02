import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

TARGET_COLUMNS = [
    'Destination Port', 'Timestamp', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count', 'Average Packet Size', 'Label'
]

def load_and_preprocess(csv_source, nrows=None, freq='10s'):
    """
    Handles minute-only timestamps by distributing intra-minute records 
    evenly across the minute to enable sub-minute aggregation.
    """
    df = pd.read_csv(csv_source, usecols=lambda c: c.strip() in TARGET_COLUMNS, nrows=nrows)
    df.columns = df.columns.str.strip()
    
    # Flexible Datetime Parsing — try known format first, then broad fallbacks
    raw_ts = df['Timestamp'].copy()
    df['Timestamp'] = pd.to_datetime(raw_ts, format='%d/%m/%Y %H:%M', errors='coerce')
    still_na = df['Timestamp'].isna()
    if still_na.sum() > 0:
        # Try without a fixed format (handles many common variants)
        df.loc[still_na, 'Timestamp'] = pd.to_datetime(
            raw_ts[still_na], dayfirst=True, errors='coerce'
        )
    still_na = df['Timestamp'].isna()
    if still_na.sum() > 0:
        # Last-resort: let pandas infer the format completely
        df.loc[still_na, 'Timestamp'] = pd.to_datetime(
            raw_ts[still_na], infer_datetime_format=True, errors='coerce'
        )

    na_count = df['Timestamp'].isna().sum()
    print(f"  Timestamp parsing: {len(df) - na_count}/{len(df)} rows parsed successfully "
          f"({na_count} rows dropped as unparseable).")
    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp')
    if len(df) == 0:
        # Sample a few raw values to help diagnose the format
        sample = raw_ts.dropna().head(5).tolist()
        raise ValueError(
            f"All timestamps failed to parse — 0 rows remain after dropna.\n"
            f"Sample raw Timestamp values: {sample}\n"
            f"Update the format string in load_and_preprocess() to match these."
        )

    # Synthesize intra-minute offset seconds for records sharing the same minute
    group_counts = df.groupby('Timestamp').transform('count')['Destination Port']
    cum_counts = df.groupby('Timestamp').cumcount()
    df['Timestamp'] = df['Timestamp'] + pd.to_timedelta((cum_counts / group_counts) * 59.9, unit='s')

    # Binary Attack Labeling
    df['Label_Binary'] = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Aggregate into discrete 10-second sub-windows
    df['window'] = df['Timestamp'].dt.floor(freq)
    
    grouped = df.groupby('window').agg(
        flow_count=('Destination Port', 'count'),
        unique_dst_ports=('Destination Port', 'nunique'),
        mean_flow_duration=('Flow Duration', 'mean'),
        total_fwd_packets=('Total Fwd Packets', 'sum'),
        total_bwd_packets=('Total Backward Packets', 'sum'),
        total_length_fwd=('Total Length of Fwd Packets', 'sum'),
        flow_bytes_per_sec=('Flow Bytes/s', 'mean'),
        flow_packets_per_sec=('Flow Packets/s', 'mean'),
        syn_count=('SYN Flag Count', 'sum'),
        rst_count=('RST Flag Count', 'sum'),
        ack_count=('ACK Flag Count', 'sum'),
        fin_count=('FIN Flag Count', 'sum'),
        mean_packet_length=('Average Packet Size', 'mean'),
        mean_iat=('Flow IAT Mean', 'mean'),
        max_iat=('Flow IAT Max', 'mean'),
        is_attack=('Label_Binary', 'max')
    ).reset_index()
    
    feature_cols = [c for c in grouped.columns if c not in ['window', 'is_attack']]
    return grouped, feature_cols

def create_sequences(state_df, feature_cols, seq_len=10, scaler=None, is_train=True):
    data = state_df[feature_cols].values
    labels = state_df['is_attack'].values
    
    if is_train:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        joblib.dump(scaler, "scaler.pkl")
        print("Successfully saved: scaler.pkl")
    else:
        if scaler is None:
            if os.path.exists("scaler.pkl"):
                scaler = joblib.load("scaler.pkl")
            else:
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data)
                joblib.dump(scaler, "scaler.pkl")
                return create_sequences(state_df, feature_cols, seq_len, scaler, is_train=True)
        scaled_data = scaler.transform(data)
        
    X_seq, y_next, y_labels = [], [], []
    for i in range(len(scaled_data) - seq_len):
        X_seq.append(scaled_data[i : i + seq_len])
        y_next.append(scaled_data[i + seq_len])
        y_labels.append(labels[i + seq_len])
        
    return np.array(X_seq), np.array(y_next), np.array(y_labels), scaler

if __name__ == "__main__":
    monday_path = "data/Monday-WorkingHours.pcap_ISCX.csv"
    
    if os.path.exists(monday_path):
        print(f"Processing dataset from: {monday_path}")
        state_df, feature_cols = load_and_preprocess(monday_path, nrows=None, freq='10s')
        X_seq, y_next, _, _ = create_sequences(state_df, feature_cols, seq_len=10, is_train=True)
        
        # Explicitly save numpy sequence tensors
        np.save("X_seq_train.npy", X_seq)
        np.save("y_next_train.npy", y_next)
        
        print(f"Preprocessing completed successfully!")
        print(f"Saved: X_seq_train.npy (Shape: {X_seq.shape})")
        print(f"Saved: y_next_train.npy (Shape: {y_next.shape})")
    else:
        print(f"Error: Target dataset file not found at '{monday_path}'")
        print("Please verify your file path inside src/preprocess.py")