from utils.nn_pt_utils import TorchClassifierWrapper, saved_model_usage
from utils.ml_utils import re_sample, pre_model
from utils.prepropcessing_utils import plot_category
from collections import Counter
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


# =======================================  PREPARE DATA  ================================================
def handling_data_pt(path=1, plot=False):
        path1 = 'data/PTT_BK_usage.csv'
        path2 = 'data/PTT_BK_usage2.csv'

        if path == 1:
            df = pd.read_csv(path1)
            df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
            df.dropna(inplace=True)
            df = df.reset_index(drop=True)
        elif path == 2:
            df = pd.read_csv(path2)
            drop_cols = ['Date','target','future_return_5','rsi_result', 'macd_result', 'bb_result', 'adx_result']
            df = df.drop(columns=drop_cols).dropna()

            df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
            df.dropna(inplace=True)
            df = df.reset_index(drop=True)
        else:
             raise "********** PATH INVALID **************"

        # features = ['Low','SMA_5', 'rsi', 'macd_diff',  'adx', 'target_encoded']
        # df = df[features]
        # print(df.head(5))
        print(Counter(df['target_encoded']))
        print(f'Shape of data: {df.shape}')
        print(df.columns)

        df_balanced = re_sample(df, y_col='target_encoded')


        if plot: 
            plot_category('target_encoded', df_balanced)

        X, y, preprocessor = pre_model(df_balanced)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

        # classifier_main_xgb(X_train, X_test, y_train, y_test, preprocessor=preprocessor)

        return X_train, X_test, y_train, y_test, preprocessor

# =======================================  PROCESSE1: TRAINING  ================================================
# Train the model
def train_model(X_train, y_train, preprocessor, get_prediction=False, plot=False):

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', TorchClassifierWrapper(hidden_dim=128, output_dim=3,  # number feature input
                                         epochs=100, lr=0.001, criteria='cross-ent',
                                         batch_size=16, val_size=0.1, patience=15, debug=False))       # number feature input
    ])


    pipeline.fit(X_train, y_train)
    print("\nModel training complete..........")


    # Get feature names after transformation
    feature_names = preprocessor.get_feature_names_out()
    print("Feature names after preprocessing:", feature_names)


    # Get predictions
    if get_prediction:
        print("\nModel get prediction..........")
        x = X_train[500:505]
        predictions = pipeline.predict(x)
        print(predictions)

    # plot performances
    if plot :
        print("\nModel plot..........")
        pipeline.named_steps['model'].plot_performance()

    return pipeline
# =======================================  PROCESSE2: EVALUATION ================================================
# Evaluation part
def evaluate_model(pipeline, X_test, y_test, X_train, y_train, cv=False):
    print("\nModel evaluation..........")

    # Cross-validation scores while training
    if cv == True:
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        print("Cross-validation scores:", scores)
        print("Average CV accuracy:", scores.mean())

    # Evaluate the Model with test set
    report, acc, precision, recall, f1, cm, cm_display = pipeline.score(X_test, y_test)
    print(report)
    print('Confusion matrix:')
    print(cm)


# =======================================  PROCESSE3: SAVE MODEL ================================================
# Saved relevant files
def save_model(pipeline, label_encoder=None):
    model_name = "ANN_pt"
    # Save the model pipeline> model> model(keras)
    pipeline.named_steps['model'].save_model(model_name)

    # Save the preprocessor
    preprocess = pipeline.named_steps['preprocess']
    joblib.dump(preprocess, f'model/{model_name}_preprocessor.pkl')

    # Save the LabelEncoder
    if label_encoder:
        joblib.dump(label_encoder, f'model/{model_name}_label_encoder.pkl')

    # 

# =======================================  PROCESSE4: GET PREDICTION ================================================
# Make Predictions and usage model:
def use_model(X, plot=False):
    path_model = "model/ANN_pt_complete.pth"
    path_his = "model/ANN_pt_history.pth"
    path_pre = "model/ANN_pt_preprocessor.pkl"
    # path_label = "model/ANN_pt_label_encoder.pkl"

    saved_model = saved_model_usage(path_model=path_model, path_his=path_his, path_pre=path_pre)
    model, history = saved_model.load_model()

    if plot:
        saved_model.plot_saved_history()    # plot model    

    # Get predictions
    print("\nModel get prediction..........")
    x = X[10:15]

    predictions = saved_model.get_prediction(x)

    return predictions


   

if __name__ == "__main__":
    # STEP1: prepare data
    X_train, X_test, y_train, y_test, preprocessor = handling_data_pt(path=2, plot=False)
    
    # STEP2: traning
    pipepline = train_model(X_train, y_train, preprocessor,get_prediction=True, plot=False)

    # STEP3: EVALUATE MODEL
    evaluate_model(pipepline, X_test, y_test, X_train, y_train, cv=False)

    # STEP4 : SAVE MODEL
    save_model(pipepline, label_encoder=None)

    # STEP5 : USAGE SAVED MODEL
    pred = use_model(X_test, plot=False)
    
    key_value = {'sell': 0, 'hold': 1, 'buy': 2}
    reverse_key = {v: k for k, v in key_value.items()}
    pred_label = [reverse_key[i] for i in pred]
    print(pred_label)

    



