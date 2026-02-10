# MambaStock: Selective State Space Model for Stock Prediction

[🇹🇼 繁體中文說明](README.zh-TW.md) | [🇺🇸 English](README.md)

**MambaStock** 實作了一個基於 **Mamba (S6)** 架構（結構化狀態空間序列模型）的股價預測模型。Mamba 在序列建模任務中取得了顯著的成功，在保持 Transformer 效能的同時，提供了線性時間複雜度。  
本儲存庫利用歷史股票數據，使用 **滑動視窗 (Sliding Window)** 方法來預測未來的價格趨勢，並包含一個專門的推論步驟，用於預測下一個交易日的價格。

## **✨ 主要功能**

* **Mamba 架構**：比 Transformer 更有效地處理長序列時間序列數據，且記憶體使用量更低。  
* **自動獲取數據**：整合 yfinance 自動下載股票數據（支援全球股票代碼，例如：2330.TW, AAPL, NVDA）。  
* **滑動視窗**：使用歷史視窗（例如：過去 20 天）來預測下一個時間步，防止前瞻偏差 (look-ahead bias)。  
* **未來推論**：在訓練後自動預測下一個交易日 (T+1) 的股價。

## **🛠️ 需求**

我們使用 uv 進行高速的依賴管理和環境設置。

### **1\. 安裝**

首先，複製 (clone) 儲存庫：  
```
git clone https://github.com/LouieLK/MambaStock.git
cd MambaStock
```

### **2\. 設置環境**

同步依賴項（如果配置正確，這將自動建立虛擬環境並安裝支援 CUDA 的 PyTorch）：  
```
uv sync
```

## **🚀 使用方法**

您可以直接使用 uv run 執行訓練腳本。該腳本會自動處理數據下載、預處理、訓練和可視化。

### **使用 CUDA (推薦)**
```
uv run python main.py --use-cuda
```
### **僅使用 CPU**
```
uv run python main.py
```
### **自定義訓練範例**

針對台積電 (2330.TW) 進行訓練，滑動視窗為 60 天：  
uv run python main.py \--ts-code 2330.TW \--seq-len 60 \--use-cuda

## **⚙️ 選項**

可以使用命令行參數來自定義模型行為。以下是可用選項的完整列表：

| 參數 | 類型 | 預設值 | 說明 |
| :---- | :---- | :---- | :---- |
| \--use-cuda | Flag | False | 啟用 CUDA 訓練（需要 NVIDIA GPU）。 |
| \--ts-code | str | 2330.TW | 股票代碼符號（例如 2330.TW, AAPL）。 |
| \--seq-len | int | 20 | 滑動視窗的大小（回溯期）。 |
| \--epochs | int | 50 | 訓練的輪數 (epochs)。 |
| \--batch-size | int | 64 | 訓練的批次大小。 |
| \--lr | float | 0.001 | 學習率。 |
| \--hidden | int | 32 | Mamba 層中隱藏狀態的維度。 |
| \--layer | int | 2 | 堆疊的 Mamba 層數。 |
| \--n-test | int | 365 | 用於測試集（回測）的天數。 |
| \--wd | float | 1e-5 | 權重衰減 (L2 正則化)。 |
| \--seed | int | 1 | 用於可重現性的隨機種子。 |

## **📚 引用**

```
@article{shi2024mamba,  
  title={MambaStock: Selective state space model for stock prediction},  
  author={Zhuangwei Shi},  
  journal={arXiv preprint arXiv:2402.18959},  
  year={2024},  
}  
```