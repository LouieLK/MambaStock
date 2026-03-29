import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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