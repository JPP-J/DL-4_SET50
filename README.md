# 📊 SET50 Stock Signal Pipeline 

A complete machine learning pipeline for analyzing **PPT Thai SET50 stock market data**, including:

- **ETL**
- **Exploratory Data Analysis (EDA)**
- **Feature Engineering**
- **Feature Selection**
- **Classification Modeling with with MLP (Multilayer Perceptron)**


## 📌 Overview

This project aims to classify `buy`, `hold`, and `sell` signals for **PTT stock** in SET50 stocks using historical data (2001-12-06 to 2025-6-01) and technical indicators.

1. **ETL (Extract, Transform, Load)** – Download and clean SET50 historical stock data  
2. **EDA** – Explore trends, distributions, outliers, , including example visualizations such as **MACD** and **Bollinger Bands (BB)** graphs  
3. **Data Cleaning** – Remove missing values, detect and remove outliers, fix skewness in features  
4. **Feature Engineering** – Generate technical indicators (MACD, RSI, BB, ADX, etc.)  
5. **Feature Selection** – Use ANOVA and manual filtering via box plots  
6. **Resampling** – Balance classes using oversampling or undersampling techniques  
7. **Model Training** – Train a classification model (e.g., PyTorch MLP)  
8. **Prediction & Evaluation** – Assess signal prediction performance  

## 📦 Library Usage

The following Python libraries are used in this project:

- `yfinance` – For downloading stock data  
- `pandas`, `numpy` – Data manipulation and numerical operations  
- `matplotlib`, `seaborn` – Data visualization  
- `ta` – Technical analysis indicators  
- `sklearn` – Feature selection and preprocessing  
- `scipy` – Statistical tests (e.g. ANOVA)  
- `PyTorch (torch)` – Deep learning model  
- `joblib` – Saving models and pipelines  
- `datetime`, `os`, etc. – Utility modules


## 📈 Feature Engineering Details

Over **25 features** are generated to enhance predictive power:
| Feature Category        | Features                                  |
|------------------------|-------------------------------------------|
| 📉 Lag Features         | Close_t-1, Close_t-2, Volume_t-1          |
| 🧮 Rolling Statistics   | SMA_5, STD_5, EMV_5                       |
| 📐 Technical Indicators | rsi, macd, macd_signal, macd_diff, <br> bb_upper, bb_lower, bb_width, <br> adx, adx_pos, adx_neg |
| 📆 Date Features        | Year, Quarter, Month, Day, Weekday        |
| 📊 Price-Based Features | Daily_Return, Close_Diff_t-1, future_return_5 |


## 🔍 Feature Selection Pipeline

1. **ANOVA Test**  
   - Retain only features with `p-value < 0.05`

2. **Box Plot Review**  
   - Visually inspect distributions and separability

3. **Save Selected Features**  
   - Final feature set saved to `data/PTT_BK_usage.csv`

*But training with full feature and selected feature to select the best performance* 


## 🧠 Model Architecture (PyTorch MLP)

```python
TorchModel(
  (fc1): Linear(29 → 128) → BatchNorm → ReLU
  (fc2): Linear(128 → 128) → BatchNorm → ReLU
  (dropout1): Dropout(0.1)
  (fc3): Linear(128 → 128) → BatchNorm → ReLU
  (dropout2): Dropout(0.2)
  (fc4): Linear(128 → 128) → BatchNorm → ReLU
  (dropout3): Dropout(0.3)
  (fc5): Linear(128 → 128) → BatchNorm → ReLU
  (fc6): Linear(128 → 3)
)
```

### 📊 Model Performance Metrics
- For overall of model test: **Accuracy  0.85**

    | Class | Precision | Recall | F1-Score | Support |
    |-------|-----------|--------|----------|---------|
    | 0     | 0.81      | 0.93   | 0.87     | 706     |
    | 1     | 0.86      | 0.67   | 0.75     | 706     |
    | 2     | 0.88      | 0.95   | 0.91     | 707     |


- Final Metrics While Training:
    - Final Train Loss: 0.3796
    - Final Val Loss: 0.4046
    - Final Train Accuracy: 0.8408
    - Final Val Accuracy: 0.8361


### 🔢 Confusion Matrix

|       | Predicted 0 | Predicted 1 | Predicted 2 |
|-------|-------------|-------------|-------------|
| True 0| 659         | 44          | 3           |
| True 1| 151         | 470         | 85          |
| True 2| 4           | 33          | 670         |


### 📈 Dataset Summary

- **Class distribution:** `Counter({1: 3531, 0: 1370, 2: 793})`  
- **Data shape:** `(5694, 30)` with full features

---

### 🔍 Features Used

| Feature         | Feature         | Feature         | Feature         | Feature         |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| Open            | High            | Low             | Close           | Volume          |
| sqrt_Volume     | Close_t-1       | Close_t-2       | Volume_t-1      | SMA_5           |
| STD_5           | EMV_5           | rsi             | macd            | macd_signal     |
| macd_diff       | bb_upper        | bb_lower        | bb_width        | adx             |
| adx_pos         | adx_neg         | Daily_Return    | Close_Diff_t-1  | Year            |
| Quarter         | Day             | Month           | Weekday         | target_encoded  |
