from datetime import datetime as dt
from utils.yf_utils import download_stock_data
import pandas as pd


# ticker = 'PTT.BK'
# a = f"{ticker.replace('.', '_')}.csv"
# print(a)


# future = {'buy': 1,
#               'hold': 0,
#               'sell': -1
#               }

# a = ['buy', 'sell', 'buy']
# a = pd.Series(a)

# print(a.map(future))


i = 3
a = i%2
print(a)
aa = ['Open', 'High', 'Low', 'Close', 'Volume', 'sqrt_Volume', 'Close_t-1', 'Close_t-2', 'Volume_t-1', 'SMA_5', 'STD_5',
 'EMV_5', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_upper', 'bb_lower', 'bb_width', 'adx', 'adx_pos', 'adx_neg', 
 'Daily_Return', 'Close_Diff_t-1', 'Year', 'Day', 'target_encoded']

print(len(aa))

a = 5
if a:
    print('c')

