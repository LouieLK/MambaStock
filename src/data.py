import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from src.yahoo import save_yahoo_to_standard_csv


def create_sliding_window(data, target, seq_length):
    """
    製作滑動視窗數據
    X: 過去 N 天的特徵
    y: 第 N+1 天的目標
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : i + seq_length]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_data(ts_code, n_test, seq_len):
    """
    讀取、清洗、切割、標準化資料
    """
    print("📂 Preparing Data...")
    # 路徑處理
    data_path = Path('data') / f"{ts_code}.csv"

    # 讀取資料
    save_yahoo_to_standard_csv(ts_code) 
        
    df = pd.read_csv(data_path)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    
    # 目標：預測 'pct_chg' (轉換為小數點格式)
    target_col = 'pct_chg'
    targets = df[target_col].apply(lambda x: 0.01 * x).values
    
    # 特徵：移除不適合的欄位
    drop_cols = ['trade_date', 'ts_code', 'pre_close', 'change', 'pct_chg']
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    features = feature_df.values
    
    # 時間切分
    split_idx = len(features) - n_test
    
    train_raw = features[:split_idx]
    test_raw = features[split_idx:]
    
    train_y_raw = targets[:split_idx]
    test_y_raw = targets[split_idx:]
    
    # 標準化 (Standardization)
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_raw)
    test_X_scaled = scaler.transform(test_raw)
    
    # 製作滑動視窗
    X_train, y_train = create_sliding_window(train_X_scaled, train_y_raw, seq_len)
    X_test, y_test = create_sliding_window(test_X_scaled, test_y_raw, seq_len)
    
    # 轉為 Tensor
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1))
    
    # 用於還原價格的輔助資料
    test_base_prices = df['close'].values[split_idx + seq_len - 1 : -1]
    test_dates = df['trade_date'].values[split_idx + seq_len:]
    
    # 防呆截斷
    min_len = min(len(test_base_prices), len(y_test))
    test_base_prices = test_base_prices[:min_len]
    test_dates = test_dates[:min_len]
    y_test = y_test[:min_len]
    
    return train_dataset, test_dataset, features.shape[1], test_dates, test_base_prices, y_test, scaler, features, df