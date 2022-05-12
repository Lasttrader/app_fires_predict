from flask import Flask, request
import numpy as np
import pandas as pd
import pickle
import json

app = Flask(__name__)

def preprocess_and_predict(value_dict):
    
    """
    предсказывает вероятность возникновения пожара по переданным показателям
    погоды в словаре с разбивкой по погодным станциям
    
    параметры: словарь значений,
    value_dict['Date'] - дата получения данных по погоде в формате день-месяц-год
    остальные ключи в словаре - погодные станции, названия которых должны совпадать
    с названиями станций, их количеством и порядком их следования из файла samara_stations
    """
    
    # количество станций в словаре
    # берем длину словаря и отнимаем 1 (элемент с датой)
    stations_amount = len(value_dict) - 1
    
    # считываем станции
    samara_stations = pd.read_csv('data/samara_stations.csv', index_col=0)[:stations_amount]
    
    # получаем список значений показателей для станций
    value_list = [list(value_dict[station].values()) for station in samara_stations.index]

    # присваиваем значение даты
    date_idx = value_dict['Date']
    
    # переводим список показателей в массив
    value_array = np.array(value_list)
    
    # формируем dataframe для предсказания
    ds = pd.DataFrame(data=value_array, 
                      columns = ['T', 'P', 'U', 'Ff', 'Td', 'RRR',
                                 'DD', 'N'], 
                      index=[date_idx for i in range(stations_amount)])
    
    # индекс переводим к типу дата
    ds.index = pd.to_datetime(ds.index, dayfirst=True) 

    # приводим к типу float те столбцы, где возможно
    for col in ds.columns:
        try:
            ds[col] = ds[col].astype('float')
        except:
            pass    
    
    # кодирование OHE
    a = pd.get_dummies(ds).copy()
    
    # добавление новых столбцов дня и месяца
    a['day'] = a.index.day # день записи
    a['month'] = a.index.month  # месяц записи

    # получаем список тренировочных столбцов, mean и std для стандартизации
    train_cols, scl_mean, scl_std = pickle.load(open('columns_and_scaling_const.pkl', 'rb'))
    # загружаем модель
    model = pickle.load(open('xgb_model.pkl', 'rb'))

    # формируем DF со всеми тренировочными столбцами
    # и значениями из gui
    df_predict = pd.DataFrame(columns=train_cols)
    for col in df_predict.columns:
        try:
            df_predict[col] = a[col]
        except:
            df_predict[col] = 0

    # стандартизация
    df_predict = (df_predict-scl_mean)/scl_std # масштабируем вектор
    
    # получаем массив вероятностей возникновеня пожара для всех станций
    prediction_prob = model.predict_proba(df_predict.values).T[1]
    
    # получаем (долготу, широту)
    lon_lat = [*zip(samara_stations['lon'].values, samara_stations['lat'].values)]

    # получаем вероятность в %
    p = [*map('{:.2%}'.format, prediction_prob)]

    # словарь для передачи
    data_dict = {'station': samara_stations.index.to_list(),
                 'lon_lat': lon_lat,
                 'P': p}

    # переводим словарь в json
    # encode() - переводим в байтовую строку, декодируем
    data_json = json.dumps(data_dict).encode().decode('unicode-escape')
    
    # возвращаем дату и json с предсказаниями
    return data_json

@app.route('/', methods=['GET', 'POST'])
def home():
    
    return 'Hello! We are forecasting fires here.'

@app.route('/service_fires', methods=['GET', 'POST'])
def service():
    
    # получаем данные от клиента
    json_data = request.json
    
    # предсказываем вероятность и получаем словарь json с прогнозом по станциям
    pred_json = preprocess_and_predict(json_data)
    
    # возвращаем строку json
    return pred_json

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=5000, debug=True)