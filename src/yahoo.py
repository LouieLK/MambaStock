import yfinance as yf
import pandas as pd

def save_yahoo_to_standard_csv(ts_code):
    """
    將 Yahoo Finance 日線 DataFrame 轉換成指定格式 CSV
    ts_code: 股票代碼，例如 '2330.TW'
    """
    stock = yf.Ticker(ts_code+'.TW')
    df = stock.history(period="max")
    name = stock.info.get('shortName') or "找不到名稱"
    print(f'🏢 股票公司名稱: {name}')
    df = df.copy()

    if df.empty:
        print(f"❌ 無法下載 {ts_code} 的資料")
        assert 0

    # 重置索引，把日期變成欄位
    df.reset_index(inplace=True)
    df.rename(columns={'Date':'trade_date',
                       'Open':'open',
                       'High':'high',
                       'Low':'low',
                       'Close':'close',
                       'Volume':'vol'}, inplace=True)
    
    df['open'] = round(df['open'],2)
    df['high'] = round(df['high'],2)
    df['low'] = round(df['low'],2)
    df['close'] = round(df['close'],2)

    # 將日期轉成 YYYYMMDD 格式
    df['trade_date'] = df['trade_date'].dt.strftime('%Y%m%d')
    
    # ts_code
    df['ts_code'] = ts_code
    
    # pre_close
    df['pre_close'] = df['close'].shift(1)
    
    # change
    df['change'] = round(df['close'] - df['pre_close'],2)
    
    # pct_chg (%)
    df['pct_chg'] = round((df['change'] / df['pre_close']) * 100,2)
    
    # amount = vol * close （假設 vol 單位為股）
    df['amount'] = round(df['vol'] * df['close'],2)
    
    # 調整欄位順序
    columns_order = ['ts_code','trade_date','open','high','low','close','pre_close',
                     'change','pct_chg','vol','amount']

    df = df[columns_order]
    
    # 去掉第一筆 (因為 shift 造成 pre_close 為 NaN)
    df = df[1:]

    # 儲存 CSV
    filename = './data/'+ts_code+'.csv'
    df.to_csv(filename, index=False)
    print(f"💾 CSV 已儲存為 {filename}")

