import numpy as np
import pandas as pd
from pandas import Timestamp
from datetime import date, timedelta, datetime

# -------------------------------------------------------------------------------------
# Fonksiyon: create_train_test_data
# Bu fonksiyon, verilen zaman serisi verilerini (df)
# pencere boyutuna (window_size) göre geciktirir ve
# belirli bir test boyutu (test_size) kadar son veriyi test seti olarak ayırır. Gün sayısı cinsinden değer alır.
# time_series_col_name parametresi, zaman serisi verisinin bulunduğu sütun adıdır.
# -------------------------------------------------------------------------------------
def create_train_test_data(df, window_size, test_size, time_series_col_name):
    """
    Parametreler:
      df: Zaman serisi verilerini içeren DataFrame.
      window_size: Her örnek için kullanılacak geçmiş veri adedi.
      test_size: Test setinde kullanılacak gün sayısı.
      time_series_col_name: Zaman serisi verisinin bulunduğu sütun adı.

    İşlem:
      1. "timestamp" sütununu datetime formatına çevirir.
      2. Test setini, son test_size kadar gün içeren verilerden oluşturur.
      3. Eğitim setini, geri kalan verilerden oluşturur.
      4. Hem eğitim hem de test seti için pencere yapısı oluşturur.

    Dönüş:
      df_x_train, df_y_train, df_x_test, df_y_test
    """
    # 1. Zaman bilgilerini datetime formatına çeviriyoruz.
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2. Test setinde kaç gün olacağını timedelta ile belirliyoruz.
    days_in_test = timedelta(days=test_size)

    # 3. Test setini son test_size gün içeren veriler olarak ayırıyoruz.
    df_test = df[df["timestamp"] > (df["timestamp"].max() - days_in_test)]
    # 4. Eğitim setini, test setine dahil olmayan veriler olarak ayırıyoruz.
    df_train = df[df["timestamp"] <= (df["timestamp"].max() - days_in_test)]

    # 5. Zaman serisi verilerinin bulunduğu sütunu numpy array olarak alıyoruz hem eğitim hem de test verisi için.
    df_test_values = df_test[time_series_col_name].values
    df_train_values = df_train[time_series_col_name].values

    # 6. Eğitim ve test verilerini pencere (trajectory) formatına dönüştürüyoruz.
    train_data_as_traj = np.zeros((len(df_train) - window_size, window_size + 1), dtype="float32")
    test_data_as_traj = np.zeros((len(df_test) - window_size, window_size + 1), dtype="float32")

    # Her pencere için, verileri kaydırarak atıyoruz.
    for i in range(window_size + 1):
        train_data_as_traj[:, i] = np.reshape(
            df_train_values[i:len(df_train_values) + i - window_size],
            len(df_train_values[i:i+len(df_train_values)-window_size])
        )
    for i in range(window_size + 1):
        test_data_as_traj[:, i] = np.reshape(
            df_test_values[i:len(df_test_values) + i - window_size],
            len(df_test_values[i:i+len(df_test_values)-window_size])
        )

    # 7. Eğitim verisini özellik (x) ve hedef (y) olarak ayırıyoruz.
    df_x_train = train_data_as_traj[:, :-1]
    df_y_train = train_data_as_traj[:, -1]

    # 8. Test verisini özellik (x) ve hedef (y) olarak ayırıyoruz.
    df_x_test = test_data_as_traj[:, :-1]
    df_y_test = test_data_as_traj[:, -1]

    return df_x_train, df_y_train, df_x_test, df_y_test