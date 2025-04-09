# TEST for best 3

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

time_series_col_name = "total_flow"

top_k = 250

window_size = 40
test_size = 84
X_train_val, y_train_val, X_test, y_test = create_train_test_data(updated_df, window_size, test_size, time_series_col_name)

train_length = 4*24*7*40 # train kümesinde 40 hafta olacak şekilde size belirleniyor.
X_train = X_train_val[-train_length:]
y_train = y_train_val[-train_length:]


w_K_combinations = [(5,40),(16,10),(38,25)]

prediction_list = []

for comb in w_K_combinations:
    w, K = comb
    print(comb)
    neighbors_indices = find_nearest_targets(X_train[:,-w:], X_test[:,-w:], K)
    one_y_pred = np.array([np.average(y_train[neighbors_indices[i].astype(int)]) for i in range(len(X_test))])
    prediction_list.append(one_y_pred)

average_of_predictions = sum(prediction_list)/len(prediction_list)

print("MAPE:",MAPE(y_test, average_of_predictions))
print("MAE:",MAE(y_test, average_of_predictions))