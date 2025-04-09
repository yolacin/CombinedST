import numpy as np
import pandas as pd
from pandas import Timestamp
from datetime import date, timedelta, datetime
from numba import njit, types, extending

# -------------------------------------------------------------------------------------
# Fonksiyon: euclidean_distance
# Numba'nın @njit dekoratörü ile hızlandırılmış Öklid mesafe hesaplama fonksiyonu.
# -------------------------------------------------------------------------------------
@njit
def euclidean_distance(v1, v2):
    # İki vektör arasındaki Öklid mesafesini hesaplar.
    return np.sqrt(np.sum((v1 - v2) ** 2))

# -------------------------------------------------------------------------------------
# Fonksiyon: weighted_euc_dist
# Ağırlıklı Öklid mesafesi hesaplaması yapar.
# -------------------------------------------------------------------------------------
@njit
def weighted_euc_dist(v1, v2, w):
    # Verilen ağırlıklar (w) ile vektör farkının karesini alıp toplamın karekökünü hesaplar.
    return np.sqrt(np.sum((w * (v1 - v2)) ** 2))

# -------------------------------------------------------------------------------------
# Fonksiyon: find_nearest_targets
# Bu fonksiyon, eğitim setindeki (df_train_x_values) her bir örnek için,
# test setindeki (df_test_x_values) örneğe en yakın top_k eğitim örneğinin indekslerini bulur.
# Numba'nın extending.overload_method dekoratörü ile yazılmıştır.
# -------------------------------------------------------------------------------------
@extending.overload_method(types.Array, 'argpartition')
def find_nearest_targets(df_train_x_values, df_test_x_values, top_k):
    """
    Parametreler:
      df_train_x_values: Eğitim setindeki özellik değerleri.
      df_test_x_values: Test setindeki özellik değerleri.
      top_k: Her test örneği için seçilecek en yakın eğitim örneği sayısı.

    İşlem:
      1. Her test örneği için, tüm eğitim örnekleriyle arasındaki ağırlıklı Öklid mesafesini hesaplar.
      2. Bu mesafelerden en küçük top_k mesafeyi veren eğitim örneklerinin indekslerini döner.
    """
    # Eğitim verisinin sütun sayısını belirliyoruz.
    num_cols = df_train_x_values.shape[1]
    length_of_train = int(len(df_train_x_values))

    # Her sütun için ağırlıkları hesaplıyoruz.
    weights = np.array([(2 * (k + 1)) / (num_cols * (num_cols + 1)) for k in range(num_cols)])

    # Sonuçları saklamak için boş bir numpy array oluşturuyoruz.
    nearest_trajectory_indicies_of_train_set = np.zeros((len(df_test_x_values), top_k), dtype=np.float32)

    # Test setindeki her örnek için döngü:
    for i in range(len(df_test_x_values)):
        # Tüm eğitim verileri ile olan mesafeleri hesaplamak için boş bir array.
        distances = np.zeros((length_of_train, ), dtype=np.float32)

        # Seçilen test örneği.
        selected_row = df_test_x_values[i]

        # Her bir eğitim örneği için mesafe hesaplaması:
        for j in range(length_of_train):
            row_j = df_train_x_values[j]
            dist = weighted_euc_dist(selected_row, row_j, weights)
            #dist = euclidean_distance(selected_row, row_j)
            distances[j] = dist

        # En küçük top_k mesafeye sahip eğitim örneklerinin indekslerini alıyoruz.
        indicies_of_nearests = np.argpartition(distances, top_k)[:top_k]

        # seçilen en yakın top_k tane trajectory de uzaklıklarına göre sıralanıyor ve indeksleri alınıyor.
        re_distances = np.zeros((top_k, ), dtype=np.float32)
        for k in range(top_k):
            re_dist = weighted_euc_dist(selected_row, df_train_x_values[indicies_of_nearests[k]], weights)
            re_distances[k] = re_dist
        dx = np.argsort(re_distances)
        indicies_of_nearests = indicies_of_nearests[dx]
        nearest_trajectory_indicies_of_train_set[i] = indicies_of_nearests

    return nearest_trajectory_indicies_of_train_set