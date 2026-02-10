import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
from pathlib import Path
from src.mamba import Mamba, MambaConfig
from src.yahoo import save_yahoo_to_standard_csv

# ==========================================
# 1. 工具函數與設定 (Utility & Config)
# ==========================================

def set_seed(seed, cuda):
    """
    設定隨機種子 確保反覆訓練結果一致
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def evaluation_metric(y_true, y_pred):
    """
    計算並輸出訓練數據
    """
    MSE = mean_squared_error(y_true, y_pred)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)
    print(f'MSE: {MSE:.4f} | RMSE: {RMSE:.4f} | MAE: {MAE:.4f} | R2: {R2:.4f}')

# ==========================================
# 2. 資料處理核心 (Data Processing)
# ==========================================

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

def prepare_data(args):
    """
    讀取、清洗、切割、標準化資料
    """
    # 路徑處理
    data_path = Path('data') / f"{args.ts_code}.csv"

    # 讀取資料
    save_yahoo_to_standard_csv(args.ts_code) 
        
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
    split_idx = len(features) - args.n_test
    
    train_raw = features[:split_idx]
    test_raw = features[split_idx:]
    
    train_y_raw = targets[:split_idx]
    test_y_raw = targets[split_idx:]
    
    # 標準化 (Standardization)
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_raw)
    test_X_scaled = scaler.transform(test_raw)
    
    # 製作滑動視窗
    X_train, y_train = create_sliding_window(train_X_scaled, train_y_raw, args.seq_len)
    X_test, y_test = create_sliding_window(test_X_scaled, test_y_raw, args.seq_len)
    
    # 轉為 Tensor
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1))
    
    # 用於還原價格的輔助資料
    test_base_prices = df['close'].values[split_idx + args.seq_len - 1 : -1]
    test_dates = df['trade_date'].values[split_idx + args.seq_len:]
    
    # 防呆截斷
    min_len = min(len(test_base_prices), len(y_test))
    test_base_prices = test_base_prices[:min_len]
    test_dates = test_dates[:min_len]
    y_test = y_test[:min_len]
    
    return train_dataset, test_dataset, features.shape[1], test_dates, test_base_prices, y_test, scaler, features, df

# ==========================================
# 3. 模型定義 (Model)
# ==========================================

class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, layers, seq_len):
        super().__init__()
        self.config = MambaConfig(d_model=hidden, n_layers=layers)
        self.embedding = nn.Linear(in_dim, hidden)
        self.mamba = Mamba(self.config)
        self.head = nn.Sequential(
            nn.Linear(hidden * seq_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.mamba(x)
        x = x.reshape(x.size(0), -1)
        x = self.head(x)
        return x

# ==========================================
# 4. 訓練與預測流程
# ==========================================

def train_model(model, train_loader, args):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()
    model.train()
    for e in range(args.epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            if args.cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            opt.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (e+1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {e+1}/{args.epochs} | Loss: {avg_loss:.6f}')

def predict(model, test_loader, args):
    """
    用於測試集的批次預測
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            if args.cuda:
                x_batch = x_batch.cuda()
            output = model(x_batch)
            predictions.append(output.cpu().numpy())
    return np.concatenate(predictions)

def predict_next_day(model, scaler, features, seq_len, device):
    """
    🔮 預測未來一天的獨立函數
    """
    # 取出最後一段數據 (Raw Data)
    last_window_raw = features[-seq_len:]
    
    # 獨立進行標準化 (Transform only)
    # 這裡使用從 prepare_data 傳出來的 scaler，確保標準化基準一致
    last_window_scaled = scaler.transform(last_window_raw)
    
    # 轉 Tensor (增加 Batch 維度 -> [1, seq_len, features])
    input_tensor = torch.FloatTensor(last_window_scaled).unsqueeze(0)
    if device:
        input_tensor = input_tensor.cuda()
    
    # 推論
    model.eval()
    with torch.no_grad():
        pred_pct = model(input_tensor).item()
        
    return pred_pct

# ==========================================
# 5. 主程式 (Main)
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='CUDA training.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Dimension of representations')
    parser.add_argument('--layer', type=int, default=2,
                        help='Num of layers')
    parser.add_argument('--n-test', type=int, default=365,
                        help='Size of test set')
    parser.add_argument('--ts-code', type=str, default='2330',
                        help='Stock code')  
    parser.add_argument('--seq-len', type=int, default=20,
                        help='size of sliding window')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='size of batch')


    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()
    
    print(f"🔧 Device: {'CUDA' if args.cuda else 'CPU'}")
    set_seed(args.seed, args.cuda)

    # 1. 準備資料
    print("📂 Preparing Data...")
    # 🌟 修改點：這裡接收 scaler, features, df
    train_dataset, test_dataset, feature_dim, test_dates, test_base_prices, y_true_pct, scaler, features, df = prepare_data(args)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"📊 Features Dimension: {feature_dim}, Sequence Length: {args.seq_len}")

    # 2. 初始化模型
    model = Net(in_dim=feature_dim, out_dim=1, hidden=args.hidden, layers=args.layer, seq_len=args.seq_len)
    if args.cuda:
        model = model.cuda()

    # 3. 訓練
    print("🚀 Start Training...")
    train_model(model, train_loader, args)

    # 4. 測試集回測
    print("📊 Backtesting on Test Set...")
    y_pred_pct = predict(model, test_loader, args).flatten()
    
    # 確保長度一致
    min_len = min(len(y_pred_pct), len(test_base_prices))
    y_pred_pct = y_pred_pct[:min_len]
    y_true_pct = y_true_pct[:min_len]
    test_base_prices = test_base_prices[:min_len]
    test_dates = test_dates[:min_len]

    pred_prices = test_base_prices * (1 + y_pred_pct)
    true_prices = test_base_prices * (1 + y_true_pct)

    print("\n📈 Evaluation Metrics (Percentage Change):")
    evaluation_metric(y_true_pct, y_pred_pct)

    # 5. 🔮 未來預測 (Future Prediction)
    print("\n" + "="*40)
    print("🔮 Forecasting Future (Next Day)...")
    
    # 呼叫獨立的預測函數
    next_pct = predict_next_day(model, scaler, features, args.seq_len, args.cuda)
    
    # 計算價格
    last_close = df['close'].iloc[-1]
    last_date = df['trade_date'].iloc[-1]
    next_price = last_close * (1 + next_pct)
    next_date = last_date + pd.Timedelta(days=1)
    
    print(f"📅 最後交易日: {last_date.date()}")
    print(f"💵 最後收盤價: {last_close:.2f}")
    print("-" * 40)
    print(f"🚀 預測下一日: {next_date.date()}")
    print(f"📈 預測漲跌幅: {next_pct*100:.2f}%")
    print(f"💰 預測股價:   {next_price:.2f}")
    print("="*40 + "\n")

    # 6. 畫圖
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, true_prices, label='Actual Price', color='black', alpha=0.7)
    plt.plot(test_dates, pred_prices, label='Predicted Price (Mamba)', color='red', alpha=0.7)
    
    # 將未來預測點畫上去 (用虛線連接)
    plt.plot([last_date, next_date], [last_close, next_price], 
             color='blue', linestyle='--', marker='o', label='Future Forecast')
    
    plt.title(f'Stock Prediction: {args.ts_code}', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()