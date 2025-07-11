from sklearn.compose import TransformedTargetRegressor
# pre-processing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

# params
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# decomposition
from sklearn.decomposition import PCA


# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # clf task
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score       # reg task

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
# Classification
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBRegressor, XGBClassifier

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pickle
import joblib
from collections import Counter
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from utils.prepropcessing_utils import plot_category, anova_check



def pre_model(df:pd.DataFrame):

    numerical_features = df.select_dtypes(include=[np.number]).columns.values
    numerical_features = numerical_features[numerical_features != 'target_encoded']

    # Defines parameters
    X = df.drop(columns='target_encoded')
    y = df['target_encoded']             # Labels:   {'likely_sell': 0, 'hold': 1, 'likely_buy': 2, 'buy': 3, 'sell': 4}

    preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), numerical_features)
    ], remainder='passthrough'  # numeric columns untouched
    )

    return X, y, preprocessor

def evaluate_clsf(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    cm_matrix = confusion_matrix(y_test, y_pred)
    return acc , clf_report, cm_matrix

def plot_confusion_matrix(cm, labels_cm, title=''):
    # plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',cbar=False,
                xticklabels=labels_cm,
                yticklabels=labels_cm)
    plt.xlabel('Predicted Label', color='red')
    plt.ylabel('True Label', color='blue')
    plt.title(f'Confusion Matrix {title}')


def classifier_main_xgb(X_train, X_test, y_train, y_test, preprocessor):

    # Define the Pipeline with the ColumnTransformer and the classifier
    pipeline_xgb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', TransformedTargetRegressor(XGBClassifier(random_state=42, max_depth=5, eval_metric='logloss')))
    ])

    # Training
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    pipeline_xgb.fit(X_train, y_train)
    print(f"Pipeline fitted: {hasattr(pipeline_xgb, 'steps')}")
    y_pred = pipeline_xgb.predict(X_test)

    # Evaluate
    acc, report, cm = evaluate_clsf(y_test=y_test, y_pred=y_pred)
    print('\nXGBoost Classifier')
    print(f'Accuracy Score: {acc:.4f} ')
    print(f"\nClassification Report:\n{report}")
    print(f"\nConfusion Matrix:\n{cm}")

    # Get unique classes for consistent ordering
    unique_classes = np.unique(y_train)
    print(unique_classes)

    # Extract the XGBoost model and feature names from the pipeline
    xgb_model = pipeline_xgb.named_steps['classifier']
    feature_names = pipeline_xgb.named_steps['preprocessor'].get_feature_names_out()

    # VISUALIZATION
    # Feature Importance
    plt.figure(figsize=(12, 8))
    # plt.barh(feature_names, xgb_model.feature_importances_)
    plt.barh(feature_names, xgb_model.regressor_.feature_importances_)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('XGBoost Feature Importance')
    plt.show()

    # Confusion Matrix
    plt.figure(figsize=(12, 8))
    plot_confusion_matrix(cm=cm, labels_cm=unique_classes, title='XGBoost Classifier')
    plt.show()

    print(pipeline_xgb)

def re_sample(df:pd.DataFrame, y_col:str='target_encoded'):

    df_majority = df[df[y_col] == 1]
    df_minority_0 = df[df[y_col] == 0]
    df_minority_2 = df[df[y_col] == 2]

    # Upsample minority
    df_minority_0_upsampled = resample(df_minority_0, replace=True, n_samples=len(df_majority), random_state=42)
    df_minority_2_upsampled = resample(df_minority_2, replace=True, n_samples=len(df_majority), random_state=42)

    # Combine
    df_balanced = pd.concat([df_majority, df_minority_0_upsampled, df_minority_2_upsampled])
    return df_balanced

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