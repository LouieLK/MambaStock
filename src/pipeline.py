import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.utils import set_seed, evaluation_metric
from src.data import prepare_data
from src.model import Net
from src.engine import train_model, predict, predict_next_day

def plot_outcome(ts_code, test_dates, true_prices, pred_prices, last_date, last_close, next_date, next_price, next_pct):

    print(f"📅 最後交易日: {last_date.date()}")
    print(f"💵 最後收盤價: {last_close:.2f}")
    print("-" * 40)
    print(f"🚀 預測下一日: {next_date.date()}")
    print(f"📈 預測漲跌幅: {next_pct*100:.2f}%")
    print(f"💰 預測股價:   {next_price:.2f}")
    print("="*40 + "\n")

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, true_prices, label='Actual Price', color='black', alpha=0.7)
    plt.plot(test_dates, pred_prices, label='Predicted Price (Mamba)', color='red', alpha=0.7)
    
    plt.plot([last_date, next_date], [last_close, next_price], 
             color='blue', linestyle='--', marker='o', label='Future Forecast')
    
    plt.title(f'Stock Prediction: {ts_code}', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def run_mamba_pipeline(params: dict):

    ts_code = params.get('ts_code', '2330.TW')
    cuda = params.get('use_cuda', False)
    seed = params.get('seed', 1)
    epochs = params.get('epochs', 50)
    lr = params.get('lr', 0.001)
    wd = params.get('wd', 1e-05)
    hidden = params.get('hidden', 32)
    layer = params.get('layer', 2)
    n_test = params.get('n_test', 365)
    seq_len = params.get('seq_len', 20)
    batch_size = params.get('batch_size', 64)
    show_plot = params.get('show_plot', False)

    set_seed(seed, cuda)
    train_dataset, test_dataset, feature_dim, test_dates, test_base_prices, y_true_pct, scaler, features, df = prepare_data(ts_code, n_test, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📊 Features Dimension: {feature_dim}, Sequence Length: {seq_len}")

    model = Net(in_dim=feature_dim, out_dim=1, hidden=hidden, layers=layer, seq_len=seq_len)
    if cuda:
        model = model.cuda()

    train_model(model, train_loader, epochs=epochs, lr=lr, wd=wd, cuda=cuda)

    # 4. 測試集回測
    print("📊 Backtesting on Test Set...")
    y_pred_pct = predict(model, test_loader, cuda=cuda).flatten()
    
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
    
    next_pct = predict_next_day(model, scaler, features, seq_len, cuda)
    
    last_close = df['close'].iloc[-1]
    last_date = df['trade_date'].iloc[-1]
    next_price = last_close * (1 + next_pct)
    next_date = last_date + pd.Timedelta(days=1)
    
    if show_plot:
        plot_outcome(ts_code, test_dates, true_prices, pred_prices, last_date, last_close, next_date, next_price, next_pct)
    
    # 6. 回傳給 API 的結果
    return {
        "ts_code": ts_code,
        "next_price": next_price,
        "next_pct": next_pct,
        "status": "success"
    }