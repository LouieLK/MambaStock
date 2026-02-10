# MambaStock: Selective State Space Model for Stock Prediction

[🇹🇼 繁體中文說明](README.zh-TW.md) | [🇺🇸 English](README.md)

**MambaStock** implements a stock price prediction model based on the **Mamba (S6)** architecture (Structured State Space Sequence Models). Mamba has achieved remarkable success in sequence modeling tasks, offering linear time complexity while maintaining the performance of Transformers.  
This repository leverages historical stock data to predict future price trends using a **Sliding Window** approach and includes a dedicated inference step for forecasting the next trading day's price.

## **✨ Key Features**

* **Mamba Architecture**: Efficiently handles long time-series data with lower memory usage than Transformers.  
* **Auto Data Fetching**: Integrated with yfinance to automatically download stock data (supports global tickers, e.g., 2330.TW, AAPL, NVDA).  
* **Sliding Window**: Uses historical windows (e.g., past 20 days) to predict the next time step, preventing look-ahead bias.  
* **Future Inference**: Automatically predicts the stock price for the upcoming trading day (T+1) after training.

## **🛠️ Requirements**

We use uv for high-speed dependency management and environment setup.

### **1\. Installation**

First, clone the repository:  
```
git clone https://github.com/LouieLK/MambaStock.git
cd MambaStock
```
### **2\. Setup Environment**

Sync dependencies (this will automatically create a virtual environment and install PyTorch with CUDA support if configured):  
```
uv sync
```
## **🚀 Usage**

You can run the training script directly using uv run. The script handles data downloading, preprocessing, training, and visualization automatically.

### **With CUDA (Recommended)**
```
uv run python main.py --use-cuda
```
### **CPU Only**
```
uv run python main.py
```
### **Custom Training Example**

Train on TSMC (2330.TW) with a 60-day sliding window:  
uv run python main.py \--ts-code 2330.TW \--seq-len 60 \--use-cuda

## **⚙️ Options**

The model behavior can be customized using command-line arguments. Here is the full list of available options:

| Argument | Type | Default | Description |
| :---- | :---- | :---- | :---- |
| \--use-cuda | Flag | False | Enable CUDA training (requires NVIDIA GPU). |
| \--ts-code | str | 2330.TW | Stock ticker symbol (e.g., 2330.TW, AAPL). |
| \--seq-len | int | 20 | Size of the sliding window (lookback period). |
| \--epochs | int | 50 | Number of training epochs. |
| \--batch-size | int | 64 | Batch size for training. |
| \--lr | float | 0.001 | Learning rate. |
| \--hidden | int | 32 | Dimension of the hidden state in Mamba layer. |
| \--layer | int | 2 | Number of Mamba layers stacked. |
| \--n-test | int | 365 | Number of days to use for the test set (backtesting). |
| \--wd | float | 1e-5 | Weight decay (L2 regularization). |
| \--seed | int | 1 | Random seed for reproducibility. |

## **📚 Citation**

```
@article{shi2024mamba,  
  title={MambaStock: Selective state space model for stock prediction},  
  author={Zhuangwei Shi},  
  journal={arXiv preprint arXiv:2402.18959},  
  year={2024},  
}  
```