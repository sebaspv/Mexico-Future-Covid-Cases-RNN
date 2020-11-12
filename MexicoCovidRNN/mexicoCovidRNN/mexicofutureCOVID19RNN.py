from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#data src = https://coronavirus.gob.mx/datos/#DOView
#download in csv

covid_cases = pd.read_csv('/Users/macbookpro/Downloads/covidmexico.csv')
covid_cases = covid_cases.transpose()
covid_cases.columns = covid_cases.iloc[2]

covid_cases = covid_cases.drop('cve_ent')
covid_cases = covid_cases.drop('nombre')
covid_cases = covid_cases.drop('poblacion')
covid_cases = covid_cases['Nacional']
#set covid cases to only the national total sum column

index = []
for i in range(len(covid_cases.index)):
  index.append(i)
  i+=1
#create new numerical index
cases_sum = []
n = 0
for i in covid_cases:
  n+=i
  cases_sum.append(n)
covid_cases = cases_sum
test_point = np.round(len(covid_cases)*.1)
#get data quantity from percentage
test_ind = int(len(covid_cases)-test_point)
train = covid_cases[:test_ind]
test = covid_cases[test_ind:]
print(len(test),'train')
print(len(train),'train')
#we divide the historical data into train and test clusters
#length of train dataset = 223
#length of test dataset = 25
scaler = MinMaxScaler()
scaler.fit(np.reshape(train,(223,1)))
scaled_train = scaler.transform(np.reshape(train,(223,1)))
scaled_test = scaler.transform(np.reshape(test,(25,1)))
#define prediction training length
pred_len = 20
#define future prediction numbers
batch_size=1
#Create training generator
generator = TimeseriesGenerator(scaled_train,scaled_train,
                                length=pred_len,batch_size=batch_size)
#Create validation generator
val_generator = TimeseriesGenerator(scaled_test,scaled_test,
                                length=pred_len,batch_size=batch_size)
#The number of features in which the data will predict the next values.
n_features = 1 

#Create the LSTM Model

model = Sequential()
model.add(LSTM(60,input_shape=(pred_len,n_features),return_sequences=True))
model.add(LSTM(60,return_sequences=True))
model.add(LSTM(30))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
#Declare an early stop
early_stop = EarlyStopping(monitor='val_loss',patience=2)
#Fit the training data into the model
model.fit(generator,epochs = 10,validation_data=val_generator,callbacks=[early_stop])
test_predictions = []

first_eval_batch = scaled_train[-pred_len:]
current_batch = first_eval_batch.reshape((1, pred_len, n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
#Create predictions 25 steps into the future

true_predictions = scaler.inverse_transform(test_predictions)
#Transform the scaled data back to the original values, since the predictions
#are scaled.
preds = pd.DataFrame(test)
preds.columns = ['True Predictions']
preds['Model Predictions'] = true_predictions
preds.plot()
plt.show()