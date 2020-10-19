from t2v_multi_amp import T2V
import pandas as pd
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import optimizers
import Helper_functions as hf
from sklearn.preprocessing.data import MinMaxScaler

pd.options.display.width = 0

def get_split(type_):
    if type_ =="raw":
        return {"sb002":72,
                "sb003":38,
                "sb006":135,
                "sb008":55}
    elif type_ == "UNI":
        return {"sb002":-1,
                "sb003":-1,
                "sb006":-1,
                "sb008":-1}
type_ = "raw"
user = 'sb006'
user_id = user + "_time_v"
df = pd.read_csv("FOLDER/"+user_id + ".csv", sep=';')

features = df.drop('pretty_time', axis=1).columns

#Data preprocessing
data = df.drop('pretty_time', axis=1).values

split = 24 * get_split(type_)[user]


train_dim = int(len(df) - split)
timesteps = 24 * 7

scaler = MinMaxScaler()
scaler = scaler.fit(data[:train_dim])

train_data = scaler.transform(data[:train_dim])
test_data = scaler.transform(data[train_dim:])

X_train, y_train = hf.func().create_dataset_m_to_one(train_data, timesteps)
X_test, y_test = hf.func().create_dataset_m_to_one(test_data, timesteps)


#Create validation data
val_dim = int(len(X_train) - split) 
X_val = X_train[val_dim:]
X_train = X_train[:val_dim]
y_val = y_train[val_dim:]
y_train = y_train[:val_dim]

#Hyperparameters
run = 9
hidden_layers = 1 
epochs_ = 300  
batch_size_ = 200  
lr_ = 0.0001      
dropout_ = 0.0   
unit_size = 75     
t2v_ = True
t2v_k = 170 #(7 * 24)

#Build model
model = Sequential()
if t2v_:
    model.add(T2V(t2v_k))
        
model.add(LSTM(units=75, input_shape=(timesteps, len(features)), use_bias=True, return_sequences=True, recurrent_dropout=0))
for _ in range(hidden_layers):
    model.add(LSTM(units=unit_size, use_bias=True, return_sequences=True))
    model.add(Dropout(rate=dropout_))

model.add(LSTM(units=unit_size, use_bias=True, return_sequences=False, recurrent_dropout=dropout_))

#Add output layer
model.add(Dense(units=len(features)))

model.compile(optimizer=optimizers.Adam(lr=lr_), loss='mse')

#Train model
history = model.fit(X_train, y_train, epochs=epochs_, batch_size=batch_size_, validation_data=(X_val, y_val))
#Make predictions
results = [model.predict(X_test), model.evaluate(X_test, y_test)]

#print(model.summary())    
hf.func().format_results_raw(results, y_test, model.metrics_names, features, scaler).to_csv(user_id + '_y_hat_' + str(run) + '.csv', index=False)
hf.plotter().plot_func(history, user_id + '_' + str(run))
pd.DataFrame(history.history["loss"], history.history["val_loss"]).to_csv(str(run)+'_history.csv')
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
