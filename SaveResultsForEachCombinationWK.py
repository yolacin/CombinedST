import numpy as np
import pandas as pd

# En iyi w ve K kombinasyonularını bulmak için aşağıdaki kodlar tasarlanmıştır.
# Verilen bir w listesi ve K listesi ve N tam sayısı için tüm kombinasyonlar oluşturulur ve generate_N_combinations_prediction
# fonksiyonu N li tüm kombinasyonların sonuçlarını dataframe olarak çıktı verir.
# Örneğin N=2 için (w_1,K_1) ve (w_2,K_2) olası tahminlerinin ortalaması alınır ve sonuçlar dataframe olarak çıkar.
# Bir başka örnek: N=3 için (w_1,K_1) ve (w_2,K_2) ve (w_3,K_3) 
# olası tahminlerinin ortalaması alınır ve sonuçlar dataframe olarak çıkar.

def preds_load(FileNum, w, K, top_k=250):
    x = np.load(file_path +"/{}_y__val_pred_w{}_K{}.npy".format(FileNum, w, K))
    return x

def generate_w_K_list(list_w,list_K):
    w_K_list = []
    for w in list_w:
        for K in list_K:
            w_K_list.append([w,K])
    return w_K_list

# -------------------------------------------------------------------------------------
# Fonksiyon: MAPE
# Gerçek ve tahmin edilen değerler arasındaki Ortalama Mutlak Yüzde Hata'yı hesaplar.
# -------------------------------------------------------------------------------------
def MAPE(Y_actual, Y_Predicted):
    # Parametreleri numpy array'e çevirip yeniden şekillendiriyoruz.
    Y_Predicted = np.array(Y_Predicted).reshape((len(Y_Predicted), 1))
    Y_actual = np.array(Y_actual).reshape((len(Y_actual), 1))
    mape = np.mean(np.abs((Y_actual - Y_Predicted) / Y_actual)) * 100
    return mape

# -------------------------------------------------------------------------------------
# Fonksiyon: MAE
# Gerçek ve tahmin edilen değerler arasındaki Ortalama Mutlak Hata'yı hesaplar.
# -------------------------------------------------------------------------------------
def MAE(Y_actual, Y_Predicted):
    # Parametreleri numpy array'e çevirip yeniden şekillendiriyoruz.
    Y_Predicted = np.array(Y_Predicted).reshape((len(Y_Predicted), 1))
    Y_actual = np.array(Y_actual).reshape((len(Y_actual), 1))
    mae = np.mean(np.abs(Y_actual - Y_Predicted))
    return mae

from itertools import combinations

def avg_predictions_in_list(preds_list):
    return sum(preds_list)/len(preds_list)

def generate_N_combinations_prediction(FileNum,N,w_range,K_range):
    w_K_list = generate_w_K_list(w_range,K_range)
    N_combinations = list(combinations(w_K_list, N))

    all_pred_array = np.zeros((len(X_val),len(w_K_list)))

    for i in range(len(w_K_list)):
        if i%100 == 0:
            print("The {} th pred list is loaded.".format(i))
        all_pred_array[:,i] = preds_load(stat_num, w_K_list[i][0], w_K_list[i][1])


    def comb_name(comb_s):
        return "_".join(["({},{})".format(item[0],item[1]) for item in comb_s])

    results = pd.DataFrame(columns=["station_num","combination","mape","mae"])

    for j in range(len(N_combinations)):
        combs = N_combinations[j]
        
        index_in_all_pred_array = []
        for item in combs:
            index_in_all_pred_array.append(w_K_list.index(item))

        temp_pred_list = []
        dic = {}
        dic["FileNum"] = [FileNum]
        dic["combination"] = [comb_name(combs)]
        for i in index_in_all_pred_array:
            temp_pred_list.append(all_pred_array[:,i])
        avg_preds = avg_predictions_in_list(temp_pred_list)
        dic["mape"] = [MAPE(y_val,avg_preds)]
        dic["mae"] = [MAE(y_val,avg_preds)]
        temp_res = pd.DataFrame(dic)
        results = pd.concat([results,temp_res],ignore_index=True)

    return results