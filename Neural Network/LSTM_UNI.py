import tensorflow as tf
from t2v_multi import T2V
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import keras.backend
import Helper_functions as hf
import keras.losses

def get_split(type_):
    if type_ == "UNI":
        return {"u15m_s3m":distributions_in_24h * 2,
                "u30m_s15m":distributions_in_24h * 2,
                "u60m_s20m":distributions_in_24h * 2}
        
def get_split_val(type_):
    if type_ == "UNI":
        return {"u15m_s3m":distributions_in_24h * 1,
                "u30m_s15m":distributions_in_24h * 1,
                "u60m_s20m":distributions_in_24h * 1}
type_ = "UNI"
interval_length = 30
step_size = 15

interval = "u" +str(interval_length) + "m_s" + str(step_size)  +"m"
user_id = type_ + "_" +interval

#One hour divided by step size == number of intervals per hour
intervals_hour = 60 / step_size
distributions_in_24h = int(intervals_hour * 24)

#df = pd.read_csv(user_id + '_ts.csv', sep=';').drop('pretty_time', axis=1)
df = pd.read_csv("PATH TO FOLDER"+user_id + ".csv", sep=',|;', engine="python")
data = df.drop('pretty_time', axis=1).values
features = df.drop('pretty_time', axis=1).columns

train_dim = int(len(df) - get_split(type_)[interval])
timesteps = int(intervals_hour * 5)
X_train, y_train = hf.func().create_dataset_m_to_one(data[:train_dim], timesteps)
X_test, y_test = hf.func().create_dataset_m_to_one(data[train_dim:], timesteps)

#Create validation data
val_dim = int(len(X_train) - get_split_val(type_)[interval]) #8 intervals in one day
X_val = X_train[val_dim:]
X_train = X_train[:val_dim]
y_val = y_train[val_dim:]
y_train = y_train[:val_dim]

#X_train, X_test = X[:train_dim], X[train_dim:]
#y_train, y_test = Y[:train_dim], Y[train_dim:]

#Hyperparameters
run = 11
hidden_layers = 1 #
epochs_ =  600  #800 -> 500
batch_size_ = 128  #32 -> 64
lr_ = 0.001      #0.001   
dropout_ = 0.6   #0.6  
unit_size = 125     # 80 --> 50
t2v_ = False
t2v_k = 128

#Build model
model = Sequential()
if t2v_:
    model.add(T2V(t2v_k))
model.add(LSTM(units=56, input_shape=(timesteps, len(features)), use_bias=True, return_sequences=True))
for _ in range(hidden_layers):
    model.add(LSTM(units=unit_size, use_bias=True, return_sequences=True))
    model.add(Dropout(rate=dropout_))

model.add(LSTM(units=unit_size, use_bias=True, return_sequences=False))
model.add(Dropout(rate=dropout_))

model.add(Dense(28, activation='relu'))
#Add output layer
model.add(Dense(units=len(features)))

model.compile(optimizer=optimizers.Adam(lr=lr_), loss='mse')

    #Train model
with tf.device('/cpu:0'):     
    history = model.fit(X_train, y_train, epochs=epochs_, batch_size=batch_size_, validation_data=(X_val, y_val))
    #Make predictions
    results = [model.predict(X_test), model.evaluate(X_test, y_test)]

#Save the output of the network, the expected value and the final RMSE value in a csv file  
hf.func().format_results(results, y_test, model.metrics_names, features).to_csv(user_id + '_y_hat_' + str(run) + '.csv', index=False)
hf.plotter().plot_func(history, user_id + '_' + str(run))

#Save the history of the loss function during training and validation in a csv file
pd.DataFrame(history.history["loss"], history.history["val_loss"]).to_csv(str(run)+'_history.csv')

#Save the configuration settings for the run in a csv file
pd.DataFrame({'run':str(run),
              'hidden layers':str(hidden_layers),
              'timesteps':str(timesteps), 
             'epochs':str(epochs_),
             'batch size':str(batch_size_),
             'Learning':str(lr_),
             'dropout':str(dropout_),
             'unit size':str(unit_size),
             'time2vec':str(t2v_),
             'time2vec k':str(t2v_k)},index=[0]).to_csv(user_id+'_'+str(run)+"_parameters.csv", sep=';', index=False)
print("Done.")
