import numpy as np
import pandas as pd
from datetime import datetime as dt
import ta # Technical Analysis Library in Python
import ta.momentum
import ta.trend
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator
from matplotlib import pyplot as plt
from scipy.stats import zscore
import seaborn as sns
import numpy as np
from sklearn.feature_selection import f_classif

def data_to_df(csv_file):
    """
    Reads a CSV file and returns a DataFrame.
    """
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')  
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    return df
def determine_adx_trend(adx_pos, adx_neg, adx_value=None):
    """
    Determine trend direction and strength based on ADX, +DI, and -DI.
    
    Returns:
    - One of: 'no_trend', 'beginning_uptrend', 'strong_uptrend', 'very_strong_uptrend',
              'beginning_downtrend', 'strong_downtrend', 'very_strong_downtrend'
    """
    if adx_value is not None:
        strength = np.select(
            [
                adx_value <= 20,
                (adx_value > 20) & (adx_value <= 25),
                (adx_value > 25) & (adx_value <= 50),
                adx_value > 50
            ],
            [
                'weak',
                'beginning',
                'strong',
                'very_strong'
            ],
            default='no'
        )

        direction = np.where(adx_pos > adx_neg, 'uptrend', 'downtrend')

        return np.core.defchararray.add(strength, "_" + direction)

    else:
        # Only direction without strength
        return np.where(adx_pos > adx_neg, 'uptrend', 'downtrend')

def determine_rsi_trend(rsi):
    return np.where(rsi > 70, 'overbought', 
                    np.where(rsi < 30, 'oversold','neutral'
                    ))

def determine_macd_trend(macd_diff):
    return  np.where((macd_diff > 0) & (macd_diff.shift(1) <= 0),'buy',
                                 np.where((macd_diff < 0) & (macd_diff.shift(1) >= 0),'sell', 'hold'))

def determine_bb_trend(df, threshold=5):
    # df['bb_result'] = np.where(df['Close'] < df['bb_lower'], 'buy_signal',
    #                            np.where(df['Close'] > df['bb_upper'], 'sell_signal', 'hold_signal'))
    # Define threshold (5% of BB width)
    bb_threshold = threshold/100

    # Calculate proximity conditions
    df['near_lower_bb'] = df['Close'] <= (df['bb_lower'] * (1 + bb_threshold))
    df['near_upper_bb'] = df['Close'] >= (df['bb_upper'] * (1 - bb_threshold))

    # Generate signals
    result = np.where(df['near_lower_bb'], 'Buy_signal',
                      np.where(df['near_upper_bb'], 'Sell_signal', 'Hold_signal'))
    
    return  result



def determine_by_future_return(df, threshold=0.03, days=5):
    df = df.copy()
    future_returns = df['Close'].shift(-days) / df['Close'] - 1         # (Close_t+n - Close_t) / Close_t

    return np.where(
        future_returns >= threshold, 'buy',
        np.where(future_returns <= -threshold, 'sell', 'hold')
    )


def determine_target(adx_result, rsi_result, macd_result, bb_result, future_return):

    adx = {'strong_uptrend': 1,
            'weak_uptrend': 0.5,
            'uptrend': 0.75,
            'no_trend_downtrend': -1,
            'weak_downtrend' : -0.5,
            'downtrend': -0.75
            }
    adx = { 
        'weak_uptrend': 0.25,
        'beginning_uptrend': 0.50,
        'strong_uptrend':0.75 ,
        'very_strong_uptrend': 1.00,
        'uptrend': 0.50,

        'weak_downtrend': -0.25,
        'beginning_downtrend': -0.50,
        'strong_downtrend': -0.75,
        'very_strong_downtrend': -1.00,
        'downtrend': -0.50}

    rsi = {'oversold': 1,
           'neutral': 0,
           'overbought': -1
           }
    macd = {'buy': 1,
            'hold': 0,
            'sell': -1
            }
    bb = {'Buy_signal': 1,
          'Hold_signal': 0,
          'Sell_signal': -1, 
          }
    future = {'buy': 1,
              'hold': 0,
              'sell': -1
              }
    adx_score = adx_result.map(adx)
    rsi_score = rsi_result.map(rsi)
    macd_score = macd_result.map(macd)
    bb_score = bb_result.map(bb)
    future_score = future_return.map(future)

    score = adx_score + rsi_score + macd_score + bb_score + future_score

    # return np.where(total_score >= buy_threshold, 'buy',                                              # Strong bullish signal
    #        np.where(total_score <= sell_threshold, 'sell',                                             # Strong bearish signal
    #                 'hold'))                                                                            # In-between, uncertain
    return np.where(score >= 1.0, 'buy', # 001.0 - -01.0 : 0.7445
        np.where(score <= -1.0, 'sell',
                'hold'))

    # return np.where((adx_result.isin(['strong_uptrend', 'weak_uptrend'])) & (rsi_result.isin(['oversold','neutral'])) & (macd_result.isin(['buy','hold'])) & (bb_result == 'Buy_signal'), 'buy',
    #          np.where((adx_result.isin(['strong_downtrend', 'weak_downtrend'])) & (rsi_result.isin(['overbought','neutral'])) & (macd_result.isin(['sell','hold'])) & (bb_result == 'Sell_signal'), 'sell', 
    #                   'hold'
    #          ))


def transfrom_features(df):
    """
    Transforms the DataFrame by adding new features.
    """
    # lag features
    df['Close_t-1'] = df['Close'].shift(1)
    df['Close_t-2'] = df['Close'].shift(2)
    df['Volume_t-1'] = df['sqrt_Volume'].shift(1)

    # Rolling Statistics
    df['SMA_5'] = df['Close'].rolling(window=5).mean()          # Simple Moving Average
    df['STD_5'] = df['Close'].rolling(window=5).std()           # Rolling standard deviation
    df['EMV_5'] = df['Close'].ewm(span=5).mean()                # Exponential Moving Average
    
    # Technical Indicators 
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()     # Relative Strength Index
    df['rsi_result'] = determine_rsi_trend(rsi=df['rsi'])

    # df['macd'] = ta.trend.MACD(df['Close']).macd()            # Moving Average Convergence Divergence 
    macd_obj = ta.trend.MACD(df['Close'])
    df['macd'] = macd_obj.macd()                                # Moving Average Convergence Divergence 
    df['macd_signal'] = macd_obj.macd_signal()
    df['macd_diff'] = macd_obj.macd_diff()                      # This is the histogram
    df['macd_result'] = determine_macd_trend(df['macd_diff'])

    # --- Bollinger Bands ---
    bb_indicator = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_lower'] = bb_indicator.bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']            # Optional: band width as volatility measure
    df['bb_result'] = determine_bb_trend(df, threshold=5)   
    df = df.drop(columns=['near_lower_bb', 'near_upper_bb'])    

    # --- ADX (Average Directional Index) ---
    adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['adx'] = adx_indicator.adx()

    df['adx_pos'] = adx_indicator.adx_pos()  # +DI
    df['adx_neg'] = adx_indicator.adx_neg()  # -DI   
    # df['adx_result'] = np.where(df['adx_pos'] > df['adx_neg'], 'uptrend', 'downtrend')
    df['adx_result'] = determine_adx_trend(adx_pos=df['adx_pos'], adx_neg=df['adx_neg'], adx_value=df['adx'])

    # Price Change / Returns
    df['Daily_Return'] = df['Close'].pct_change()               # (Close_t - Close_t-1) / Close_t-1
    # df['Close_Diff'] = df['Close'].diff()                       # too much current can lead to data leak
    df['Close_Diff_t-1'] = df['Close'].diff().shift(1)          # historical 
    df['future_return_5'] = determine_by_future_return(df, threshold=0.03, days=5)

    # Date Features
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday                       # Monday=0

    # Create new target
    df['target'] = determine_target(adx_result=df['adx_result'], rsi_result=df['rsi_result'], macd_result=df['macd_result'],
                                    bb_result=df['bb_result'], future_return=df['future_return_5'])

    # print('\nMissing values before filling:')
    # print(df.isna().sum())

    # df.fillna(df.mean(), inplace=True)                          # Fill NaN values with column means
    df.dropna()

    # print('\nMissing values after filling:')
    # print(df.isna().sum())

    return df

def plot_data(col_name:str, df:pd.DataFrame):
    plt.figure(figsize=(8,6))
    plt.hist(df[col_name], bins=30, zorder=2, color='skyblue', edgecolor='black', alpha=1.0)
    plt.title(f'{col_name} Distribution')
    plt.xlabel(col_name)
    plt.ylabel('Frequency')
    plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
    plt.axvline(df[col_name].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean', zorder=3)
    plt.axvline(df[col_name].median(), color='blue', linestyle='dashed', linewidth=1, label='Median', zorder=3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_macd(col_name_y:list, col_name_hist:str, col_name_x:str, df:pd.DataFrame):
    plt.figure(figsize=(8,6))
    for col in col_name_y:
        plt.plot(df[col_name_x], df[col], label=col)

    plt.bar(df[col_name_x], df[col_name_hist], label=col_name_hist, 
            alpha=0.6, color='green', width=0.8)
    
    # Add zero line for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    title = f'MACD: {", ".join(col_name_y)} and {col_name_hist} vs {col_name_x}'
    plt.title(title)
    plt.xlabel(col_name_x)
    plt.ylabel('MACD Values')
    plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bb(col_name_y:list, col_name_hist:str, col_name_x:str, df:pd.DataFrame):
    plt.figure(figsize=(8,6))
    for col in col_name_y:
        plt.plot(df[col_name_x], df[col], label=col)

    plt.bar(df[col_name_x], df[col_name_hist], label=col_name_hist, 
            alpha=0.5, color='skyblue', width=0.8)
    
    # Add zero line for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    title = f'Bollinger Bands (BB): {", ".join(col_name_y)} and {col_name_hist} vs {col_name_x}'
    plt.title(title)
    plt.xlabel(col_name_x)
    plt.ylabel('BB Values')
    plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_correlaton_mt(df:pd.DataFrame):
    corr_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    return corr_matrix

def plot_category(col_name:str, df:pd.DataFrame):
    plt.figure(figsize=(8,6))
    # Get value counts for the categorical column
    category_counts = df[col_name].value_counts()
    
    # Create bar plot
    plt.bar(category_counts.index, category_counts.values, color='skyblue')
    
    plt.title(f'{col_name} Distribution')
    plt.xlabel(col_name)
    plt.ylabel('Count Values')
    plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_box(df:pd.DataFrame, cols=list, n_col=2, figsize=(10,6)):
    
    # Filter out non-numeric columns (including boolean)
    cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
    print(cols)

    total_cols = len(cols)

    if total_cols == 0:
        print("No numeric columns to plot.")
        return
    
    n_col = n_col
    n_row = (total_cols + n_col - 1) // n_col

    fig, axs = plt.subplots(n_row, n_col, figsize=figsize)

    axs = axs.reshape(-1)

    for i, col in enumerate(cols):
        ax = axs[i]
        groups = [df[df['target'] == val][col].dropna() for val in df['target'].unique()]
        ax.boxplot(groups, labels=df['target'].unique())
        # ax.grid(True)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')  # only horizontal grid lines
        ax.set_title(col)
    
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)
    
    plt.legend()
    plt.tight_layout()
    plt.show()




def outlier(df, col_name:str):
    print("outlier.....")

    # Calculate z-scores for the entire column
    df['z_score'] = zscore(df[col_name])
    df['z_score'] = zscore(df[col_name].dropna())
    threshold = 3
    outliers = df[abs(df['z_score']) > threshold]
    print(f'outliers from z-score: {outliers.shape}')

    new_df = df[abs(df['z_score']) <= threshold].copy()
    new_df.drop(columns=['z_score'], inplace=True)
    print(f'Original shape: {df.shape}')
    print(f'Current shape: {new_df.shape}\n')

    return new_df

def anova_check(X, y, cols_name:list):
    F, p = f_classif(X, y)
    use_ful_feature = []
    for i, j in enumerate(p):
        if j < 0.05:
            use_ful_feature.append(cols_name[i])
    return use_ful_feature
