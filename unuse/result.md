# BGBOOST
## REMOVE FEARTURES:

features selection.............................
/home/jpp/projects/04_set50/.venv/lib/python3.12/site-packages/sklearn/feature_selection/_univariate_selection.py:111: RuntimeWarning: divide by zero encountered in divide
  f = msb / msw

After Anova from 30 to 27 
can_use_features: ['Open', 'High', 'Low', 'Close', 'Volume', 'sqrt_Volume', 'Close_t-1', 'Close_t-2', 'Volume_t-1', 'SMA_5', 'STD_5', 'EMV_5', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_upper', 'bb_lower', 'bb_width', 'adx', 'adx_pos', 'adx_neg', 'Daily_Return', 'Close_Diff_t-1', 'Year', 'Day', 'target_encoded']

Lasted features after box plot: from 27 to 21 : 
['Open', 'High', 'Low', 'Close', 'sqrt_Volume', 'Close_t-1', 'Close_t-2', 'Volume_t-1', 'SMA_5', 'EMV_5', 'rsi', 'macd_diff', 'bb_upper', 'bb_lower', 'adx', 'adx_pos', 'adx_neg', 'Close_Diff_t-1', 'Year', 'Day', 'target_encoded']
Saved CSV completed !!
/home/jpp/projects/04_set50/utils/prepropcessing_utils.py:298: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  plt.legend()
Counter({1: 3531, 0: 1370, 2: 793})
Shape of data: (5694, 21)
Index(['Open', 'High', 'Low', 'Close', 'sqrt_Volume', 'Close_t-1', 'Close_t-2',
       'Volume_t-1', 'SMA_5', 'EMV_5', 'rsi', 'macd_diff', 'bb_upper',
       'bb_lower', 'adx', 'adx_pos', 'adx_neg', 'Close_Diff_t-1', 'Year',
       'Day', 'target_encoded'],
      dtype='object')
/home/jpp/projects/04_set50/utils/prepropcessing_utils.py:298: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  plt.legend()
Pipeline fitted: True

XGBoost Classifier
Accuracy Score: 0.9037 

Classification Report:
              precision    recall  f1-score   support

           0       0.89      0.94      0.91      1049
           1       0.92      0.79      0.85      1064
           2       0.91      0.98      0.94      1065

    accuracy                           0.90      3178
   macro avg       0.90      0.90      0.90      3178
weighted avg       0.90      0.90      0.90      3178


Confusion Matrix:
[[ 988   60    1]
 [ 122  841  101]
 [   6   16 1043]]
[0 1 2]
## FULL FEATURES:
============================== full feature ===================================
features selection.............................
/home/jpp/projects/04_set50/.venv/lib/python3.12/site-packages/sklearn/feature_selection/_univariate_selection.py:111: RuntimeWarning: divide by zero encountered in divide
  f = msb / msw

After Anova from 30 to 27 
can_use_features: ['Open', 'High', 'Low', 'Close', 'Volume', 'sqrt_Volume', 'Close_t-1', 'Close_t-2', 'Volume_t-1', 'SMA_5', 'STD_5', 'EMV_5', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_upper', 'bb_lower', 'bb_width', 'adx', 'adx_pos', 'adx_neg', 'Daily_Return', 'Close_Diff_t-1', 'Year', 'Day', 'target_encoded']

Lasted features after box plot: from 27 to 21 : 
['Open', 'High', 'Low', 'Close', 'sqrt_Volume', 'Close_t-1', 'Close_t-2', 'Volume_t-1', 'SMA_5', 'EMV_5', 'rsi', 'macd_diff', 'bb_upper', 'bb_lower', 'adx', 'adx_pos', 'adx_neg', 'Close_Diff_t-1', 'Year', 'Day', 'target_encoded']
Saved CSV completed !!
/home/jpp/projects/04_set50/utils/prepropcessing_utils.py:298: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  plt.legend()
Counter({1: 3531, 0: 1370, 2: 793})
Shape of data: (5694, 30)
Index(['Open', 'High', 'Low', 'Close', 'Volume', 'sqrt_Volume', 'Close_t-1',
       'Close_t-2', 'Volume_t-1', 'SMA_5', 'STD_5', 'EMV_5', 'rsi', 'macd',
       'macd_signal', 'macd_diff', 'bb_upper', 'bb_lower', 'bb_width', 'adx',
       'adx_pos', 'adx_neg', 'Daily_Return', 'Close_Diff_t-1', 'Year',
       'Quarter', 'Day', 'Month', 'Weekday', 'target_encoded'],
      dtype='object')
/home/jpp/projects/04_set50/utils/prepropcessing_utils.py:298: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  plt.legend()
Pipeline fitted: True

XGBoost Classifier
Accuracy Score: 0.9091 

Classification Report:
              precision    recall  f1-score   support

           0       0.91      0.95      0.93      1049
           1       0.93      0.79      0.86      1064
           2       0.90      0.99      0.94      1065

    accuracy                           0.91      3178
   macro avg       0.91      0.91      0.91      3178
weighted avg       0.91      0.91      0.91      3178


Confusion Matrix:
[[ 992   57    0]
 [ 100  845  119]
 [   3   10 1052]]
[0 1 2]

# NN RESULTS
## 128-3layer relu 1 2 3-dropout(0.2) 4 
           precision    recall  f1-score   support

           0       0.77      0.88      0.82       706
           1       0.79      0.61      0.69       706
           2       0.86      0.92      0.89       707

    accuracy                           0.81      2119
   macro avg       0.81      0.81      0.80      2119
weighted avg       0.81      0.81      0.80      2119

Confusion matrix:
[[624  72  10]
 [176 432  98]
 [ 14  42 651]]

 ##  128-3layer relu 1 2-dropout(0.1) 3-dropout(0.2) 4 5  
 STOP AT 82 if pateince at 15
 warnings.warn(
              precision    recall  f1-score   support

           0       0.74      0.90      0.81       706
           1       0.80      0.53      0.64       706
           2       0.83      0.93      0.88       707

    accuracy                           0.79      2119
   macro avg       0.79      0.79      0.77      2119
weighted avg       0.79      0.79      0.77      2119

Confusion matrix:
[[637  60   9]
 [211 373 122]
 [ 18  34 655]]

if patient 20
e current warning.
  warnings.warn(
              precision    recall  f1-score   support

           0       0.79      0.86      0.82       706
           1       0.77      0.61      0.68       706
           2       0.82      0.91      0.86       707

    accuracy                           0.79      2119
   macro avg       0.79      0.79      0.79      2119
weighted avg       0.79      0.79      0.79      2119

Confusion matrix:
[[607  78  21]
 [158 428 120]
 [  8  53 646]]

## 128-3layer relu 1 2 3-dropout(0.2) 4 5 

 warnings.warn(
              precision    recall  f1-score   support

           0       0.80      0.89      0.84       706
           1       0.80      0.63      0.70       706
           2       0.85      0.93      0.89       707

    accuracy                           0.81      2119
   macro avg       0.81      0.81      0.81      2119
weighted avg       0.81      0.81      0.81      2119

Confusion matrix:
[[626  66  14]
 [157 443 106]
 [  3  47 657]]


 ## 128-3layer relu 1 2 3-dropout(0.2) 4 5 6
 AT REMOVE FEATIRES
 warnings.warn(
              precision    recall  f1-score   support

           0       0.80      0.90      0.85       706
           1       0.80      0.63      0.70       706
           2       0.85      0.93      0.88       707

    accuracy                           0.82      2119
   macro avg       0.82      0.82      0.81      2119
weighted avg       0.82      0.82      0.81      2119

Confusion matrix:
[[634  64   8]
 [152 443 111]
 [  5  47 655]]

 ##  THE BEST : 128-3layer relu 1 2 3-dropout(0.2) 4 5 6

 AT FULL FEATUREA
  warnings.warn(
              precision    recall  f1-score   support

           0       0.81      0.93      0.86       706
           1       0.88      0.64      0.74       706
           2       0.87      0.97      0.92       707

    accuracy                           0.85      2119
   macro avg       0.85      0.85      0.84      2119
weighted avg       0.85      0.85      0.84      2119

Confusion matrix:
[[659  45   2]
 [156 451  99]
 [  3  18 686]]


Model Architecture:
TorchModel(
  (fc1): Linear(in_features=29, out_features=128, bias=True)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout1): Dropout(p=0.1, inplace=False)
  (fc3): Linear(in_features=128, out_features=128, bias=True)
  (bn3): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout2): Dropout(p=0.2, inplace=False)
  (fc4): Linear(in_features=128, out_features=128, bias=True)
  (bn4): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout3): Dropout(p=0.3, inplace=False)
  (fc5): Linear(in_features=128, out_features=128, bias=True)
  (bn5): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc6): Linear(in_features=128, out_features=3, bias=True)
)



## 128-3layer gelu 1 2 3-dropout(0.2) 4 5 6
AT GELU atop at 75
if add more pateince upto 20
  warnings.warn(
              precision    recall  f1-score   support

           0       0.80      0.91      0.85       706
           1       0.82      0.62      0.71       706
           2       0.84      0.93      0.88       707

    accuracy                           0.82      2119
   macro avg       0.82      0.82      0.81      2119
weighted avg       0.82      0.82      0.81      2119

Confusion matrix:
[[640  51  15]
 [160 441 105]
 [  5  48 654]]

## 128-3layer silu 1 2 3-dropout(0.2) 4 5 6
AT SILU


 ## 128- 3 layer - add label smoothing :

  warnings.warn(
              precision    recall  f1-score   support

           0       0.75      0.89      0.81       706
           1       0.78      0.54      0.64       706
           2       0.82      0.92      0.87       707

    accuracy                           0.78      2119
   macro avg       0.78      0.78      0.77      2119
weighted avg       0.78      0.78      0.77      2119

Confusion matrix:
[[629  67  10]
 [197 380 129]
 [ 18  38 651]]



64 - 3 layers
    precision    recall  f1-score   support

           0       0.72      0.87      0.79       706
           1       0.68      0.51      0.59       706
           2       0.80      0.81      0.81       707

    accuracy                           0.73      2119
   macro avg       0.73      0.73      0.73      2119
weighted avg       0.73      0.73      0.73      2119

Confusion matrix:
[[617  69  20]
 [215 363 128]
 [ 28 103 576]]