import tensorflow as tf
from t2v_multi import T2V
import pandas as pd
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import optimizers
import Helper_functions as hf


user_id = 'sb006'
interval = "u6h_s3h" 
format_ = interval + '_time'

df = pd.read_csv("PATH TO FOLDER"+user_id +'_ts_'+format_+".csv", sep=';')
features = df.drop('pretty_time', axis=1).columns

def get_split_test(type_):
    if type_ =="raw":
        return {"sb002":72,
                "sb003":38,
                "sb006":135,
                "sb008":55}
    else:
        return {"sb002":-1,
                "sb003":-1,
                "sb006":169,
                "sb008":69}
            
def get_split_val(type_):
    if type_ =="raw":
        return {"sb002":72,
                "sb003":38,
                "sb006":135,
                "sb008":55}
    else:
        return {"sb002":-1,
                "sb003":-1,
                "sb006":101,
                "sb008":41}

X, Y = [], []
#Data preprocessing
distributions_in_24h = 8
data = df.drop('pretty_time', axis=1).values

train_dim = int(len(df) - (distributions_in_24h * get_split_test(interval)[user_id]))
timesteps = int(distributions_in_24h * 7  * 2)
X_train, y_train = hf.func().create_dataset_m_to_one(data[:train_dim], timesteps)
X_test, y_test = hf.func().create_dataset_m_to_one(data[train_dim:], timesteps)

#Create validation data
val_dim = int(len(X_train) - (distributions_in_24h * get_split_val(interval)[user_id])) #intervals in one day
X_val = X_train[val_dim:]
X_train = X_train[:val_dim]
y_val = y_train[val_dim:]
y_train = y_train[:val_dim]

#Hyperparameters
run = 3
hidden_layers = 1 
epochs_ =  500  
batch_size_ = 32  
lr_ = 0.001         
dropout_ = 0.6     
unit_size = 125
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
#The CPU was faster than the GPU on the system used during the experiment
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
