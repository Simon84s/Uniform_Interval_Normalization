from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

class func:
    
    def __init__(self):
        None
        
    def format_results(self, results, test_Y, metrics_columns, features):
        
        metrics = pd.DataFrame([results[1]])
        metrics.columns = metrics_columns
       
        test_Y = pd.DataFrame(test_Y, columns=features)
        y_hat_col = []
        for col in features:
            y_hat_col.append('y_hat_' + col)
                
        y_hat = results[0]
        y_hat = pd.DataFrame(y_hat, columns=y_hat_col)
                        
        y_hat = pd.concat([test_Y, y_hat, metrics], axis=1)
        
        return y_hat.reset_index(drop=True)
        
    def format_results_m(self, results, test_Y, metrics_columns, features, timesteps):
        
        metrics = pd.DataFrame([results[1]])
        metrics.columns = metrics_columns
        
        test_Y = pd.DataFrame(test_Y.reshape(timesteps*len(test_Y), len(features)), columns=features)
                
        y_hat_col = []
        for col in features:
            y_hat_col.append('y_hat_' + col)
                
        y_hat = results[0]
        y_hat = pd.DataFrame(y_hat.reshape(timesteps*len(y_hat), len(y_hat_col)), columns=y_hat_col)
                        
        y_hat = pd.concat([test_Y, y_hat, metrics], axis=1)
        
        return y_hat.reset_index(drop=True)
    
    def format_results_raw(self, results, test_Y, metrics_columns, features, scaler):
        
        metrics = pd.DataFrame([results[1]])
        metrics.columns = metrics_columns
       
        test_Y = pd.DataFrame(scaler.inverse_transform(test_Y), columns=features)
        y_hat_col = []
        for col in features:
            y_hat_col.append('y_hat_' + col)
                
        y_hat = scaler.inverse_transform(results[0])
        y_hat = pd.DataFrame(y_hat, columns=y_hat_col)
                        
        y_hat = pd.concat([test_Y, y_hat, metrics], axis=1)
        
        return y_hat.reset_index(drop=True)
    
    def create_dataset_m_to_one(self, data, window_size):
        train, test = list(), list()
                
        for i in range(len(data)):
            in_end = i + window_size
                        
            if in_end < len(data):
                train.append(data[i:in_end])
                test.append(data[in_end])
                        
        return np.array(train), np.array(test)
   
  
    def create_dataset_m_to_m(self, data, window_size_train, window_size_predict, step):
    
        train, labels = list(), list()
                
        for i in range(0, len(data), step):
            train_end = i + window_size_train
            labels_end = train_end + window_size_predict
                    
            if labels_end < len(data):
                train.append(data[i:train_end])
                labels.append(data[train_end:labels_end])
                        
        return np.array(train), np.array(labels)
    
    def rmse_eval(self, y_hat, Y):
        
        
        return np.sqrt(np.mean(np.square(y_hat - Y), axis=-1))
    

class plotter:
    
    def __init__(self):
        None
    
    def plot_func(self, data, save_path):
                        
        #Clear plot
        plt.clf()
        
        # summarize history for loss
        plt.plot(data.history['loss'])
        plt.plot(data.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], bbox_to_anchor=(1.0, 1.0))
        plt.savefig(save_path + '_l.png')
         
    def plot_func_no_val(self, data, save_path):
                                
        #Clear plot
        plt.clf()
        
        # summarize history for loss
        plt.plot(data.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], bbox_to_anchor=(1.0, 1.0))
        plt.savefig(save_path + '_l.png')
    
    
    def plot_df(self, df, columns):
        
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True) # labels along the bottom edge are off
        
        
        x = range(len(df.values))
                
        color_ = ['black', 'orange']
        i=0
        for col in columns:
                
            plt.plot(x, df[col], color=color_[i])
            i += 1
        return plt
    
    def format_test(self, data, timestep):    
        Y = list()
        
        in_start = 0        
        for _ in range(len(data)):
        # define the end of the input sequence
            in_end = in_start + timestep
            # ensure we have enough data for this instance
            if in_end <= len(data):
                Y.append(data[in_start:in_end])
            # move along one time step
            in_start += timestep
        
        return np.array(Y)