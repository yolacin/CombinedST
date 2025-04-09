import numpy as np
import pandas as pd

# Validation verisi üzerinden en iyi w ve K değerleri için sonuçlar elde edilir.
# Seçilen w aralığı ve K aralığından elde eldilen tüm kombinasyonlar için
# tahminler .npy olarak kaydedilir ve MAE ve MAPE değerleri .csv dosyasına yazdırılır.

def save_preds_results(WinRange,KRange,top_k,PathToSave,X_train,X_val,y_train,y_val,FileNumToSave):

    """
    Fonksiyonun girdileri:
    WinRange: w lar için oluşturulmuş bir listedir.
    KRange: K değerleri için oluşturulmuş bir listedir.
    top_k: Kaç tane en yakın gezinge seçimidir.
    PathToSave: Tahminleri ve mae ve mape sonuçlarının yazıldığı dosyanın kaydedileceği adres.
    X_train: Benzer gezingelerin aranacağı eğitim verisi.
    y_train: Benzer gezingelerin bir sonraki (aday) değerleri
    X_val: Validation verisindeki gezingeler.
    y_val: Validation verisindeki output değerleri. 
    Fonksiyonun sonucu/işlevi (return): Tahminlerin kaydedilmesi
    ve tahminlerin gerçek değerlerle oluşturduğu mae ve mape değerlerinin csv dosyasına kaydedilmesi.    
    """

    results = pd.DataFrame(columns=[FileNumToSave,"W","K","mape","mae","mape_w","mae_w"])

    for w in WinRange:
    neighbors_indices = find_nearest_targets(X_train[:,-w:], X_val[:,-w:], top_k)

      for K in KRange:
          temp_indicies = neighbors_indices[:,:K].astype(int)
          y_pred_w = np.array([np.average(y_train[temp_indicies[i]], weights=1/np.arange(1, K+1)) for i in range(len(X_val))])
          y_pred = np.array([np.average(y_train[temp_indicies[i]]) for i in range(len(X_val))])

          mape_value = MAPE(y_val, y_pred)
          mae_value = MAE(y_val, y_pred)
          #print(mape_value,mae_value)
          np.save(PathtoSave + "/{}_y__val_pred_w{}_K{}.npy".format(FileNumToSave,w,K),y_pred)

          mape_value_w = MAPE(y_val, y_pred_w)
          mae_value_w = MAE(y_val, y_pred_w)
          #print(mape_value_w,mae_value_w)
          np.save(PathToSave + "/{}_y__val_weighted_pred_w{}_K{}.npy".format(FileNumToSave,w,K),y_pred_w)

          dic = {}
          dic[FileNumToSave] = [FileNumToSave]
          dic["W"] = [w]
          dic["K"] = [K]
          dic["mape"] = [mape_value]
          dic["mae"] = [mae_value]
          dic["mape_w"] = [mape_value_w]
          dic["mae_w"] = [mae_value_w]

          print(dic)
          temp_df = pd.DataFrame(dic)
          results = pd.concat([results,temp_df])
          #print(results)
          results.to_csv(PathToSave + "/{}_validation_results.csv".format(FileNumToSave))
          print("-------------------------------")

