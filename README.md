# Pronóstico del Tiempo usando redes neuronales recursivas LSTM
  Este proyecto de código abierto consiste en una red neuronal recursiva de tipo *Long Short Term Memory* (LSTM) entrenada para el pronóstico de determinadas condiciones climáticas en el Edo. Nueva Esparta, Venezuela.
  
## Introducción
  Es una red neuronal recursiva LSTM desarrollada en Python y Tensorflow, para el desarrollo de pronósticos de determinadas variables meteorológicas futuras de una región empleando datos meteorológicos históricos para desarrollar. Orientado a ser una herramienta de bajos costos y requerimientos que pueda proporcionar pronósticos climáticos de alta precisión y eficiencia que pueda ser de utilidad para organizaciones o individuos que lo requieran. 

## Funcionamiento
  El proyecto fue desarrollado utilizando **Python 3.11.8** en formato de Jupyter Notebook (.ipynb) el cual permite la e
  Para la ejecución del entrenamiento y el desarrollo de pronóstico, es necesario la instalación de las siguientes dependencias:
```Python
!pip install openmeteo_requests   #Desarrollo de solicitudes en la API
!pip install requests_cache       #Almacenamiento de Caché de requests
!pip install retry_requests       #Reintentar solicitudes de API  
!pip install pickle               #Almacenamiento y carga de modelos entrenados
```
Asimismo, es necesario instalar las mismas dependencias y otras (ya integradas en Python) para la administración de los datos, formateo de datos, construcción del modelo LSTM y el cálculo de métricas de prueba:
```python
import pickle
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from retry_requests import retry
```
  Los datos históricos meteorológicos son obtenidos de la API de datos históricos [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api#hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,rain,snowfall,surface_pressure,cloud_cover&daily=&timezone=America%2FSao_Paulo), cuyos parámetros requeridos son:
  - Latitud y Longitud de la región.
  - Periodo de tiempo.
  - Variables meteorológicas deseadas dentro del formato deseado (Por día o por hora)
  - Formato horario.
```Python
# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 10.9645415,
	"longitude": -64.0975447,
	"start_date": "1960-01-01",
	"end_date": "2024-01-01",
	"hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation", "pressure_msl", "surface_pressure", "cloud_cover", "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "soil_temperature_0_to_7cm"],
	"timezone": "America/Sao_Paulo"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°E {response.Longitude()}°N")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(4).ValuesAsNumpy()
hourly_pressure_msl = hourly.Variables(5).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(6).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(7).ValuesAsNumpy()
hourly_et0_fao_evapotranspiration = hourly.Variables(8).ValuesAsNumpy()
hourly_vapour_pressure_deficit = hourly.Variables(9).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(10).ValuesAsNumpy()
hourly_soil_temperature_0_to_7cm = hourly.Variables(11).ValuesAsNumpy()


)}
```
  Los datos históricos obtenidos son formateados y estructurados en un *Dataframe* de *Pandas*. Construyendo la secuencia temporal que funcionará como índice, es decir, la columna que indica la hora, día, més y año de cada medición de cada variable meteorológica:
```python
hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s"),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
hourly_data["Temperatura"] = hourly_temperature_2m
hourly_data["Humedad_rel"] = hourly_relative_humidity_2m
hourly_data["Punto_Dew"] = hourly_dew_point_2m
hourly_data["Temperatura_Ap"] = hourly_apparent_temperature
hourly_data["Precipitacion"] = hourly_precipitation
hourly_data["Presion_mar"] = hourly_pressure_msl
hourly_data["Presion_sup"] = hourly_surface_pressure
hourly_data["Nubosidad"] = hourly_cloud_cover
hourly_data["et0_Evot"] = hourly_et0_fao_evapotranspiration
hourly_data["Deficit_VP"] = hourly_vapour_pressure_deficit
hourly_data["Velocidad_viento"] = hourly_wind_speed_10m
hourly_data["Temp_t_superf"] = hourly_soil_temperature_0_to_7cm

hourly_dataframe = pd.DataFrame(data = hourly_data)
hourly_dataframe.head()
```
| date                | Temperatura | Humedad_rel | Punto_Dew | Temperatura_Ap | Precipitacion | Presion_mar | Presion_sup | Nubosidad | et0_Evot    | Deficit_VP | Velocidad_viento | Temp_t_superf |
|---------------------|-------------|-------------|-----------|----------------|---------------|-------------|-------------|-----------|-------------|------------|------------------|---------------|
| 1960-01-01 03:00:00 | 25.582499   | 87.39807    | 23.332499 | 26.68434       | 0.0           | 1012.4      | 1011.7057   | 54.300003 | 0.04412513  | 0.4129293  | 31.487139        | 26.6325       |
| 1960-01-01 04:00:00 | 25.432499   | 87.64929    | 23.2325   | 26.70621       | 0.1           | 1012.1      | 1011.4055   | 49.800003 | 0.04080542  | 0.40111923 | 29.899511        | 26.2825       |
| 1960-01-01 05:00:00 | 25.332499   | 87.37604    | 23.082499 | 26.721674      | 0.0           | 1011.8      | 1011.10535  | 40.5      | 0.039934024 | 0.40757418 | 28.496904        | 26.082499     |
| 1960-01-01 06:00:00 | 25.182499   | 87.89323    | 23.0325   | 26.628077      | 0.0           | 1011.7      | 1011.0052   | 30.600002 | 0.036853448 | 0.38741136 | 27.908709        | 25.832499     |
| 1960-01-01 07:00:00 | 24.932499   | 88.943016   | 22.9825   | 26.453176      | 0.0           | 1010.9      | 1010.205    | 27.3      | 0.031484142 | 0.3485973  | 27.193705        | 25.6325       |

  Asímismo, para límitar el tiempo de entrenamiento y procesamiento de datos, se limita la cantidad de variables meteorológicas a emplear para el desarrollo del modelo predictivo. Por lo que seleccionaron las siguientes variables:
- Temperatura
- Humedad relativa (Humedad_rel)
- Presión a nivel del mar (Presion_mar)
- Presión superficial (Presion_sup)
- Fáctor de evotranspiración (et0_Evot)
- Déficit de presión de vapor (Deficit_VP)

  Organizados en otro *Dataframe*, de la siguiente manera:
```python
experimental_df=hourly_dataframe.drop(labels=["Temperatura_Ap","Precipitacion","Punto_Dew","Nubosidad","Velocidad_viento","Temp_t_superf"], axis=1)
experimental_df=experimental_df.set_index("date")
experimental_df.head()
```
| date               | Temperatura | Humedad_rel | Presion_mar | Presion_sup | et0_Evot | Deficit_VP |
|--------------------|-------------|-------------|-------------|-------------|----------|------------|
| 01/01/2024   22:00 | 28.583      | 60.830189   | 1011.5      | 1010.813171 | 0.16633  | 1.530595   |
| 01/01/2024   23:00 | 27.882999   | 64.942772   | 1011.900024 | 1011.211365 | 0.11024  | 1.315266   |
| 02/01/2024   00:00 | 27.733      | 65.918678   | 1012.700012 | 1012.010559 | 0.099861 | 1.267515   |
| 02/01/2024   01:00 | 27.532999   | 66.694046   | 1013.599976 | 1012.90918  | 0.084459 | 1.224309   |
| 02/01/2024   02:00 | 27.483      | 66.683922   | 1013.799988 | 1013.10907  | 0.081843 | 1.221115   |

  Al ya tener los datos que se emplearán en el entrenamiento del modelo, se desarrollarán variables exógenas a partir de las mismas, las cuales le proporcionen al modelo la capacidad de identificar una mayor cantidad de patrones dentro de los datos. Primeramente, el modelo solamente observa valores numéricos, por lo que en la columna *date* debe ser descompueta en sus valores primarios, es decir, "hora", "dia", "mes" y "año". También se utilizará el método *day_of_week* para obtener qué día de la semana fue cada fecha. Donde 0 sería lunes y 6 sería domingo. 

```python
experimental_df['year'] = [experimental_df.index[i].year for i in range(len(experimental_df))]
experimental_df['month'] = [experimental_df.index[i].month for i in range(len(experimental_df))]
experimental_df['day'] = [experimental_df.index[i].day for i in range(len(experimental_df))]
experimental_df['hour'] = [experimental_df.index[i].hour for i in range(len(experimental_df))]
experimental_df['day_of_week'] = [experimental_df.index[i].dayofweek for i in range(len(experimental_df))]
experimental_df.reset_index(drop=True)
experimental_df.head()
```
| date               | Temperatura | Humedad_rel | Presion_mar | Presion_sup | et0_Evot | Deficit_VP | year | month | day | hour | day_of_week |
|--------------------|-------------|-------------|-------------|-------------|----------|------------|------|-------|-----|------|-------------|
| 01/01/1960   03:00 | 25.582499   | 87.398071   | 1012.400024 | 1011.705688 | 0.044125 | 0.412929   | 1960 | 1     | 1   | 3    | 4           |
| 01/01/1960   04:00 | 25.432499   | 87.649292   | 1012.099976 | 1011.405518 | 0.040805 | 0.401119   | 1960 | 1     | 1   | 4    | 4           |
| 01/01/1960   05:00 | 25.332499   | 87.376038   | 1011.799988 | 1011.105347 | 0.039934 | 0.407574   | 1960 | 1     | 1   | 5    | 4           |
| 01/01/1960   06:00 | 25.182499   | 87.893227   | 1011.700012 | 1011.005188 | 0.036853 | 0.387411   | 1960 | 1     | 1   | 6    | 4           |
| 01/01/1960   07:00 | 24.932499   | 88.943016   | 1010.900024 | 1010.205017 | 0.031484 | 0.348597   | 1960 | 1     | 1   | 7    | 4           |
| 01/01/1960   08:00 | 24.8825     | 87.867691   | 1010.799988 | 1010.105103 | 0.03597  | 0.381372   | 1960 | 1     | 1   | 8    | 4           |

Finalmente, se desarrollan variables exógenas retardadas, las cuales le indicarán al modelo el valor de cada variable en el pasado en lapsos de tiempos determinados. En este caso, las variables retardadas serán en lapsos de 7 días, 30 días y 365 días. Para permitir escalabilidad, en caso de que se emplean más variables en el entrenamiento, este proceso es desarrollado automáticamente, de la siguiente manera.

```python
lagged_features_df = pd.DataFrame()
lagged_feature = pd.DataFrame()
lagged_feature_names = []
columns=experimental_df.drop(labels=["year","month","day","hour","day_of_week"], axis=1).columns
print(columns)
lag_steps=[7, 30, 365] #In days
for y in lag_steps:
  lag_number=y*24
  for i in columns:
    lagged_feature_name = "retardado_"+str(y)+"_"+str(i)
    lagged_feature_names.append(lagged_feature_name)
    lagged_feature[lagged_feature_name] = experimental_df[i].shift(lag_number)
    lagged_features_df=pd.concat([lagged_features_df, lagged_feature]).dropna()


cualidades_exogenas=(lagged_feature_names + ["year","month","day","day_of_week","hour", 'Humedad_rel', 'Punto_Dew', 'Precipitacion', 'et0_Evot',
       'Deficit_VP'])
lagged_features_df.head()
```
| date               | retardado_7_Temperatura | retardado_7_Humedad_rel | retardado_7_Presion_mar | retardado_7_Presion_sup | retardado_7_et0_Evot | retardado_7_Deficit_VP | retardado_30_Temperatura | retardado_30_Humedad_rel | retardado_30_Presion_mar | retardado_30_Presion_sup | retardado_30_et0_Evot | retardado_30_Deficit_VP | retardado_365_Temperatura | retardado_365_Humedad_rel | retardado_365_Presion_mar | retardado_365_Presion_sup | retardado_365_et0_Evot | retardado_365_Deficit_VP |
|--------------------|-------------------------|-------------------------|-------------------------|-------------------------|----------------------|------------------------|--------------------------|--------------------------|--------------------------|--------------------------|-----------------------|-------------------------|---------------------------|---------------------------|---------------------------|---------------------------|------------------------|--------------------------|
| 31/12/1960   03:00 | 24.832499               | 82.168571               | 1015.900024             | 1015.201355             | 0.063934             | 0.558916               | 26.3825                  | 83.608162                | 1009.900024              | 1009.209229              | 0.058346              | 0.563104                | 25.582499                 | 87.398071                 | 1012.400024               | 1011.705688               | 0.044125               | 0.412929                 |
| 31/12/1960   04:00 | 24.532499               | 83.399551               | 1015.700012             | 1015.000916             | 0.057514             | 0.511105               | 26.2325                  | 83.844246                | 1010.099976              | 1009.408813              | 0.053853              | 0.550115                | 25.432499                 | 87.649292                 | 1012.099976               | 1011.405518               | 0.040805               | 0.401119                 |
| 31/12/1960   05:00 | 24.282499               | 85.437782               | 1015.299988             | 1014.600403             | 0.050028             | 0.441694               | 26.032499                | 84.330475                | 1009.400024              | 1008.708618              | 0.048444              | 0.527305                | 25.332499                 | 87.376038                 | 1011.799988               | 1011.105347               | 0.039934               | 0.407574                 |
| 31/12/1960   06:00 | 24.1325                 | 86.47374                | 1014.799988             | 1014.100525             | 0.043056             | 0.406602               | 25.832499                | 84.820663                | 1008.900024              | 1008.208679              | 0.045948              | 0.504812                | 25.182499                 | 87.893227                 | 1011.700012               | 1011.005188               | 0.036853               | 0.387411                 |

  Acto seguido, las variables retardadas son incorporadas en el *dataframe* principal. Con una cantidad total de 29 columnas o características que serán usadas en el entrenamiento del modelo LSTM.
```python
model_df=experimental_df.join(lagged_features_df).dropna()
model_df.index=pd.DatetimeIndex(model_df.index, freq=model_df.index.inferred_freq)
model_df.tail()
```
| date               | Temperatura | Humedad_rel | Presion_mar | Presion_sup | et0_Evot | Deficit_VP | year | month | day | hour | ... | retardado_30_Presion_mar | retardado_30_Presion_sup | retardado_30_et0_Evot | retardado_30_Deficit_VP | retardado_365_Temperatura | retardado_365_Humedad_rel | retardado_365_Presion_mar | retardado_365_Presion_sup | retardado_365_et0_Evot | retardado_365_Deficit_VP |
|--------------------|-------------|-------------|-------------|-------------|----------|------------|------|-------|-----|------|-----|--------------------------|--------------------------|-----------------------|-------------------------|---------------------------|---------------------------|---------------------------|---------------------------|------------------------|--------------------------|
| 01/01/2024   22:00 | 28.583      | 60.830189   | 1011.5      | 1010.813171 | 0.16633  | 1.530595   | 2024 | 1     | 1   | 22   | ... | 1011.200012              | 1010.513367              | 0.157331              | 1.371822                | 27.032999                 | 68.466713                 | 1011.5                    | 1010.809692               | 0.145719               | 1.125739                 |
| 01/01/2024   23:00 | 27.882999   | 64.942772   | 1011.900024 | 1011.211365 | 0.11024  | 1.315266   | 2024 | 1     | 1   | 23   | ... | 1011.400024              | 1010.712402              | 0.114848              | 1.216026                | 26.083                    | 76.286781                 | 1012.200012               | 1011.507019               | 0.071226               | 0.800496                 |
| 02/01/2024   00:00 | 27.733      | 65.918678   | 1012.700012 | 1012.010559 | 0.099861 | 1.267515   | 2024 | 1     | 2   | 0    | ... | 1012                     | 1011.311035              | 0.10327               | 1.094078                | 26.333                    | 74.93853                  | 1013.200012               | 1012.506714               | 0.068762               | 0.858578                 |
| 02/01/2024   01:00 | 27.532999   | 66.694046   | 1013.599976 | 1012.90918  | 0.084459 | 1.224309   | 2024 | 1     | 2   | 1    | ... | 1012                     | 1011.311279              | 0.092128              | 1.102625                | 26.083                    | 76.754669                 | 1014.099976               | 1013.405701               | 0.064118               | 0.784693                 |
| 02/01/2024   02:00 | 27.483      | 66.683922   | 1013.799988 | 1013.10907  | 0.081843 | 1.221115   | 2024 | 1     | 2   | 2    | ... | 1012.299988              | 1011.610779              | 0.088114              | 1.050719                | 25.733                    | 78.842461                 | 1014.599976               | 1013.904358               | 0.056191               | 0.699586                 |

  Es en este punto, que es posible desarrollar el preprocesamiento de datos necesarios para el entrenamiento. El cual consiste en la normalización de los valores a partir de su escala. El desarrollo de las secuencias y las etiquetas a partir de lo cual se dividen los datos entre el entrenamiento y la validación, donde el 80% de los valores son para entrenamiento y el resto para su correspondiente validación.
  Asimismo, en este punto es donde se indica la variable específica que se desea que el modelo pronostique mediante la indicación de su indice de columna en el *dataframe*. En este ejemplo, la variable a pronosticar es la Temperatura cuyo indice de columna es 0, ya que es la primera.

```python
from sklearn.preprocessing import MinMaxScaler

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(model_df)

# Define sequence length and features
sequence_length = 24  # Number of time steps in each sequence
num_features = len(model_df.columns)

# Create sequences and corresponding labels
sequences = []
labels = []
for i in range(len(scaled_data) - sequence_length):
    seq = scaled_data[i:i+sequence_length]
    label = scaled_data[i+sequence_length][0]  # 'temp' column index
    sequences.append(seq)
    labels.append(label)

# Convert to numpy arrays
sequences = np.array(sequences)
labels = np.array(labels)

# Split into train and test sets
train_size = int(0.8 * len(sequences))
rain_train_x, rain_test_x = sequences[:train_size], sequences[train_size:]
rain_train_y, rain_test_y = labels[:train_size], labels[train_size:]

print("Model_df",model_df.shape)

print("Train X shape:", rain_train_x.shape)
print("Train Y shape:", rain_train_y.shape)

print("Test X shape:", rain_test_x.shape)
print("Test Y shape:", rain_test_y.shape)
```
  La estructura de los datos de entrenamiento y prueba, así como la columnas y la cantidad de valores son las siguientes.
```python
Model_df (552288, 29)
Train X shape: (441811, 24, 29)
Train Y shape: (441811,)
Test X shape: (110453, 24, 29)
Test Y shape: (110453,)
```

  Acto seguido, se desarrola la estructura del modelo LSTM de pronostico. Importando las librerías necesarias de *tensorflow*.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Create the LSTM model
rain_model = Sequential()

# Add LSTM layers with dropout
rain_model.add(LSTM(units=128, input_shape=(rain_train_x.shape[1], rain_train_x.shape[2]), return_sequences=True))
rain_model.add(Dropout(0.2))

rain_model.add(LSTM(units=64, return_sequences=True))
rain_model.add(Dropout(0.2))

rain_model.add(LSTM(units=32, return_sequences=False))
rain_model.add(Dropout(0.2))

# Add a dense output layer
rain_model.add(Dense(units=1))

# Compile the model
rain_model.compile(optimizer='adam', loss='mean_squared_error')

rain_model.summary()
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm_3 (LSTM)               (None, 24, 128)           80896     
                                                                 
 dropout_3 (Dropout)         (None, 24, 128)           0         
                                                                 
 lstm_4 (LSTM)               (None, 24, 64)            49408     
                                                                 
 dropout_4 (Dropout)         (None, 24, 64)            0         
                                                                 
 lstm_5 (LSTM)               (None, 32)                12416     
                                                                 
 dropout_5 (Dropout)         (None, 32)                0         
                                                                 
 dense_1 (Dense)             (None, 1)                 33        
                                                                 
=================================================================
Total params: 142753 (557.63 KB)
Trainable params: 142753 (557.63 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
  Es así, que se termina de construir el modelo LSTM y es posible empezar con el entrenamiento. Para ello, se emplea el *early_stopping* para detener el entrenamiento del modelo cuando no haya mejoramiento en su rendimiento. De la siguiente manera:

```python
# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=False)

# Train the model
rain_history = rain_model.fit(
    rain_train_x, rain_train_y,
    epochs=110,
    batch_size=64,
    validation_split=0.2,  # Use part of the training data as validation
    callbacks=[early_stopping]
)
```
  Iniciando la secuencia de entrenamieno, la cual retorna el desempeño del modelo en su entrenamiento por cada iteración (epochs), hasta que el model alcanza la cantidad maxima de epochs o no mejora su rendimiento de entrenamiento.
```python
Epoch 1/110
5523/5523 [==============================] - 248s 44ms/step - loss: 9.0932e-04 - val_loss: 1.7212e-04
Epoch 2/110
5523/5523 [==============================] - 208s 38ms/step - loss: 1.9393e-04 - val_loss: 8.9977e-05
Epoch 3/110
5523/5523 [==============================] - 202s 37ms/step - loss: 1.6210e-04 - val_loss: 1.4259e-04
Epoch 4/110
5523/5523 [==============================] - 192s 35ms/step - loss: 1.5121e-04 - val_loss: 1.1684e-04
Epoch 5/110
5523/5523 [==============================] - 226s 41ms/step - loss: 1.4735e-04 - val_loss: 1.2315e-04
Epoch 6/110
5523/5523 [==============================] - 230s 42ms/step - loss: 1.4269e-04 - val_loss: 1.2994e-04
Epoch 7/110
5523/5523 [==============================] - 239s 43ms/step - loss: 1.4030e-04 - val_loss: 9.5506e-05
Epoch 8/110
5523/5523 [==============================] - 214s 39ms/step - loss: 1.3637e-04 - val_loss: 8.4449e-05
Epoch 9/110
5523/5523 [==============================] - 211s 38ms/step - loss: 1.3664e-04 - val_loss: 7.0829e-05
Epoch 10/110
5523/5523 [==============================] - 208s 38ms/step - loss: 1.3474e-04 - val_loss: 9.4878e-05
Epoch 11/110
5523/5523 [==============================] - 207s 37ms/step - loss: 1.3290e-04 - val_loss: 8.2318e-05
Epoch 12/110
5523/5523 [==============================] - 208s 38ms/step - loss: 1.3241e-04 - val_loss: 1.1044e-04
Epoch 13/110
...
Epoch 53/110
5523/5523 [==============================] - 205s 37ms/step - loss: 1.1892e-04 - val_loss: 6.3701e-05
Epoch 54/110
5523/5523 [==============================] - 204s 37ms/step - loss: 1.1837e-04 - val_loss: 6.9661e-05
```
  Al concluir el entrenamiento, es posible almacenarlo en un archivo de tipo .pkl mediante la librería de *pickle*. Para ser cargado en cualquier momento y realizar un pronóstico. 
```python
model_pkl_file = "modelo_temperatura.pkl"

with open(model_pkl_file, 'wb') as file:
    pickle.dump(rain_model, file)
```
  Seguidamente, para confirma el correcto almacenamiento del modelo, este se carga de donde fue almacenado y es sobreescrito en la variable original. Asimsimo, después de haber entrenado el modelo, se puede ejecutar el código desde este punto para realizar pronósticos sin tener que volver a entrenar el modelo. 

```python
model_pkl_file = "modelo_temperatura.pkl"

with open(model_pkl_file, 'rb') as file:  
    rain_model = pickle.load(file)
```
  Posterior al entrenamiento, es posible graficar la evolución del entrenamiento del modelo para observar cómo se desempeño y procesó los datos proporcionados.
  
```python
plt.plot(rain_history.history['loss'])
plt.plot(rain_history.history['val_loss'])
plt.title('Entrenamiento del modelo de temperatura LSTM')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
plt.show()
```
![imagen](https://github.com/isakimfv/PronosticoTiempoLSTM/assets/165188656/a848f72c-a4c1-4df6-95b9-78177aff7171)

  Es ahora, que es posible desarollar predicciones futuras de la variable tempertura. Primeramente, para evaluar el desempeño del modelo, se estimarán los valores correspondientes al 20% de los valores originales, los cuales no fueron empleados en el entrenamiento, de manera que se pueda comparar la estimación del modelo y los valores reales. A partir de lo cual se calculan las siguientes métricas de desempeño.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error


print(" ---------- Cálculo de métricas de rendimiento ---------- ")
# Predict temperatures using the trained model
predictions = rain_model.predict(rain_test_x)

# Calculate evaluation metrics
mae = mean_absolute_error(rain_test_y, predictions)
mse = mean_squared_error(rain_test_y, predictions)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

 ---------- Cálculo de métricas de rendimiento ---------- 
3452/3452 [==============================] - 32s 9ms/step
Mean Absolute Error (MAE): 0.01914467936979022
Mean Squared Error (MSE): 0.0010403882243283737
Root Mean Squared Error (RMSE): 0.032255049594263126
```

  Seguidamente, se pueden desarollar las predicciones del modelo tanto sobre los valores de validación como valores futuros más alla de los comprendidos en la llamada de API original, es decir, estimar los valores futuros.

```python
future_hours=90*24
row_count=len(model_df.columns)
# y_true values
test_y_copies = np.repeat(rain_test_y.reshape(-1, 1), rain_test_x.shape[-1], axis=-1)
true_temp = scaler.inverse_transform(test_y_copies)[:,0]

# predicted values
prediction = rain_model.predict(rain_test_x)
prediction_copies = np.repeat(prediction, row_count, axis=-1)
predicted_temp = scaler.inverse_transform(prediction_copies)[:,0]

# predicted future values
prediction = rain_model.predict(rain_test_x[-future_hours:])
prediction_copies = np.repeat(prediction, row_count, axis=-1)
predicted_future_temp = scaler.inverse_transform(prediction_copies)[:,0]
```
  A partir de los valores futuros estimados por el modelo, se conforma un nuevo *dataframe* que contiene los valores futuros organizados mediante un indice de tipo *date* que indica el día y hora de la estimación.

```python
predict_period_dates = pd.date_range(list(model_df.index)[-1], periods=future_hours, freq="h")
print(predict_period_dates)
df_forecast=pd.DataFrame({"date":predict_period_dates,"Temperatura":predicted_future_temp})
df_forecast["date"]=pd.to_datetime(df_forecast["date"])
df_forecast=df_forecast.set_index("date", drop=True)
```

  Finalmente, se grafican los valores estimados sobre la validación para comparar gráficamente la estimación y los valores reales.

```python
plt.figure(figsize=(10, 6))
plt.plot(model_df["Temperatura"].index[-77:], true_temp[-77:], label='Real')
plt.plot(model_df["Temperatura"].index[-77:], predicted_temp[-77:], label='Pronóstico')
plt.title('Pronóstico de Temperatura vs Valores reales')
plt.xlabel('Tiempo')
plt.ylabel('Temperatura')
plt.legend()
plt.show()
```
![imagen](https://github.com/isakimfv/PronosticoTiempoLSTM/assets/165188656/b0b0e6cf-25bb-4d08-a281-d4eb9cc48a12)

Y se grafican los valores futuros ajuntados con los últimos valores reales de la temperatura. Para evaluar la capacidad de continuación de la secuencia del modelo.

```python
plt.figure(figsize=(20, 4))
plt.plot(model_df["Temperatura"][-77:], label='Real')
plt.plot(df_forecast[8:150], label='Pronóstico')
plt.title('Pronóstico de Temperatura futura')
plt.xlabel('Tiempo')
plt.ylabel('Temperatura')
plt.legend()
plt.show()
```
![imagen](https://github.com/isakimfv/PronosticoTiempoLSTM/assets/165188656/a14b7b3e-b036-44da-ae94-5df951be9b1d)

