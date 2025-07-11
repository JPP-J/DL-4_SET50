from datetime import datetime as dt
from utils.yf_utils import download_stock_data
from utils.prepropcessing_utils import data_to_df, transfrom_features, plot_data, outlier, plot_macd, get_correlaton_mt, plot_bb, plot_box
from utils.prepropcessing_utils import plot_category, anova_check
from utils.ml_utils import pre_model, classifier_main_xgb, re_sample
import pandas as pd
import numpy as np 
from scipy.stats import skew
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.utils import resample



def extract_save_ticker():
    """
    Extracts the ticker symbol from a string and appends '.BK' if not present.
    """
    tickers = ['ADVANC', 'AOT', 'AWC', 'BANPU', 'BBL', 'BCP', 'BDMS', 'BEM', 'BH', 'BJC', 'BTS', 'CBG', 'CCET', 'COM7', 'CPALL', 'CPF', 'CPN', 'CRC', 'DELTA', 'EGCO', 'GPSC', 'GULF', 'HMPRO', 'IVL', 'KBANK', 'KKP', 'KTB', 'KTC', 'LH', 'MINT', 'MTC', 'OR', 'OSP', 'PTT', 'PTTEP', 'PTTGC', 'RATCH', 'SCB', 'SCC', 'SCGP', 'TCAP', 'TIDLOR', 'TISCO', 'TLI', 'TOP', 'TRUE', 'TTB', 'TU', 'VGI', 'WHA']

    for ticker in tickers:
        ticker = ticker + '.BK' 
        download_stock_data(ticker=ticker, start_date='2001-12-06', end_date='2025-6-01', interval='1d', save_to_csv=True)

# =======================================  PROCESSE1: ETL and EDA process for SET50 tickers ================================================
def data_preprocessing(file_path = 'data/PTT_BK.csv', plot=False, show_detail=False):
    print('data preprocessing.............................')
    # STEP2: Load the saved CSV files into DataFrames
    df = data_to_df(file_path)       
    if show_detail:
        print(f"Data for PTT_BK loaded successfully. Number of rows: {len(df)}")
        print(df.tail(10))

    # STEP3: Plotting the data
    if plot : 
        columns = df.columns.tolist()
        for col in columns:
            if show_detail:
                print(f"Column: {col}")
                print(df[col].describe())
                print("\n")

            plot_data(col, df)
    

    # STEP4: Remove outlier and fixing skewness data
    new_df = outlier(df, col_name='Volume')
    new_df['sqrt_Volume'] = np.sqrt(new_df['Volume'])

    if show_detail:
        print('After remove outlier and fixing skewness')
        print(skew(df['Volume']))         # Before
        print(skew(new_df['sqrt_Volume'])) # After

        if plot:
            plot_data(df=new_df, col_name='Volume')
      

    # STEP5: Transform feaurtures/Feaure Engineering 
    new_df = transfrom_features(new_df)
    if show_detail:
        print("Transformed DataFrame with new features:")
        print(new_df.head(10))

    # STEP6: example MACD chart and BB chart
    if plot:
        # MACD
        col_y = ['macd', 'macd_signal' ]
        col_hist =  'macd_diff'
        col_x = 'Date'
        plot_macd(col_name_y=col_y, col_name_hist=col_hist, col_name_x=col_x, df=df[500:600])

        # BB
        col_y = ['bb_lower', 'bb_upper', 'Close']
        col_hist =  'bb_width'
        col_x = 'Date'
        plot_bb(col_name_y=col_y, col_name_hist=col_hist, col_name_x=col_x, df=df[500:600])

        plot_category('target',  df)

    return new_df
# =======================================  PROCESSE2: Feature Selection ================================================
def features_selecting(df, plot=False, save=False):
    print('features selection.............................')

    # STEP1: select feature through ANOVA test only p-vale < 0.05 then choose
    df['target_encoded'] = LabelEncoder().fit_transform(df['target'])

    drop_cols = ['Date','target','future_return_5','rsi_result', 'macd_result', 'bb_result', 'adx_result']
    df_cleaned = df.drop(columns=drop_cols).dropna()

    columns = df_cleaned.columns.tolist()
    X = df_cleaned[columns]
    y = df_cleaned['target_encoded']

    can_use_features = anova_check(X, y, cols_name=columns)
    print(f'\nAfter Anova from {len(columns)} to {len(can_use_features)} \ncan_use_features: {can_use_features}')

    # STEP2: Recheck box plot each feature after ANOVA to select fetaure mean are different or mixed overlap and low outlier
    if plot:
        for i in range(0, len(can_use_features), 4):
            plot_box(df, cols=can_use_features[i:i+4], n_col=4, figsize=(10, 6))

    # after manual check 
    after_box = ['Volume', 'STD_5', 'macd', 'macd_signal', 'bb_width', 'Daily_Return','Closr_diff_t-1']
    can_use_features = [i for i in can_use_features if i not in after_box]
    print(f"\nLasted features after box plot: from 27 to {len(can_use_features)} : \n{can_use_features}")

    # Saved files to easier access 
    if save:
        file_name = 'data/PTT_BK_usage.csv'
        df[can_use_features].to_csv(file_name)

        file_name2 = 'data/PTT_BK_usage2.csv'
        df.to_csv(file_name2)
        print('Saved CSV completed !!')

    return df 

# =======================================  TEST WITH XGBOOST ================================================
def test_xgboost(path=1):
        path1 = 'data/PTT_BK_usage.csv'
        path2 = 'data/PTT_BK_usage2.csv'

        if path == 1:
            df = pd.read_csv(path1)
        elif path == 2:
            df = pd.read_csv(path2)
            drop_cols = ['Date','target','future_return_5','rsi_result', 'macd_result', 'bb_result', 'adx_result']
            df = df.drop(columns=drop_cols).dropna()

        df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
        df.dropna(inplace=True)
        df = df.reset_index(drop=True)

        # features = ['Low','SMA_5', 'rsi', 'macd_diff',  'adx', 'target_encoded']
        # df = df[features]
        # print(df.head(5))
        print(Counter(df['target_encoded']))
        print(f'Shape of data: {df.shape}')
        print(df.columns)

        df_balanced = re_sample(df, y_col='target_encoded')

        # df_majority = df[df['target_encoded'] == 1]
        # df_minority_0 = df[df['target_encoded'] == 0]
        # df_minority_2 = df[df['target_encoded'] == 2]

        # # Upsample minority
        # df_minority_0_upsampled = resample(df_minority_0, replace=True, n_samples=len(df_majority), random_state=42)
        # df_minority_2_upsampled = resample(df_minority_2, replace=True, n_samples=len(df_majority), random_state=42)

        # # Combine
        # df_balanced = pd.concat([df_majority, df_minority_0_upsampled, df_minority_2_upsampled])
        plot_category('target_encoded', df_balanced)

        X, y, preprocessor = pre_model(df_balanced)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

        classifier_main_xgb(X_train, X_test, y_train, y_test, preprocessor=preprocessor)

        # features = ['Open', 'High', 'Low', 'Close', 'log_Volume', 'sqrt_Volume', 'Close_t-1', 'Close_t-2', 
        #             'Volume_t-1', 'SMA_5', 'EMV_5', 'rsi', 'macd_diff', 'bb_upper', 'bb_lower', 'adx', 'adx_pos', 
        #             'adx_neg', 'Close_Diff_t-1', 'Year', 'Day', 'target_encoded']
        # features = ['Open', 'High', 'Low', 'Close', 'log_Volume', 'Close_t-1'
        #             , 'SMA_5', 'EMV_5', 'rsi', 'macd_diff', 'bb_lower', 'adx', 'adx_pos', 
        #             'adx_neg', 'target_encoded']



if __name__ == "__main__":


    # # PROCESSE1: ETL and EDA process for SET50 tickers
    # STEP1: Extract and save stock data for all SET50 tickers : 2001-12-06' to '2025-6-01'
    # extract_save_ticker()    

    
    # STEP2-5: Load the saved CSV files into DataFrames
    file_path = 'data/PTT_BK.csv'
    df = data_preprocessing(file_path=file_path, plot=False, show_detail=False)

    # PROCESSE2: Feature Selection
    features_selecting(df, plot=False, save=True)
    
    plot_category('target', df)

    
    key = {'likely_sell': 0, 'hold': 1, 'likely_buy': 2, 'buy': 3, 'sell': 4}
    count = {1: 3531, 0: 1370, 2: 793} 
    
    
    test_xgboost(path=1)


   
