import numpy as np
import pandas as pd
import glob
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

pd.options.display.width = 0

EPSILON = 1e-10

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

"""
# Mean Arctangent Absolute Percentage Error
# Metric based on radians in the unit circle (i.e., geometrically motivated)
"""
def maape(y_true, y_pred):
    
    return np.mean(np.arctan(np.abs((y_true - y_pred) / (y_true-EPSILON))))

def de_norm(user_list, dir_, out_dir, format_):
    """
	#The norm_values are the values used during normalization with uniform interval normalization
	#In the current implementation the file with the values are obtained by running norm.py (in Preprocessing folder)
	"""
	norm_values = pd.read_csv("norm_values.csv", sep=',|;', engine="python")
    
    tag = "y_hat_"
    
    #Order in which glob sorts filenames (unix ls -U sorting)
    run = [1, 10 ,2, 3, 4, 5, 6, 7, 8, 9]
    
    for ft_ in format_:
        values_ = norm_values.loc[norm_values["format"] == ft_]
        for user in user_list:
            i = 0
            for filename in glob.glob(dir_ + ft_ +'/' + user + '/*.csv'):
                if "dn" not in filename:
                    df = pd.read_csv(filename).drop("loss", axis=1)
                        
                for col in df.columns:
                
                    if "Time" not in col:
                        a = int(values_.loc[values_["features"] == col.replace(tag, '')].norm_value.values)
                        df[col] = df[col] * a
    
                df.to_csv(out_dir+ft_+'/' +user +'/dn_'+user+'_'+str(run[i]) + '.csv', index=False, sep=';')
                i += 1

def calc_rmse (user_list, dir_, format_, out_dir, normalized_):

    for ft_ in format_:
        for user in user_list:
            results = []
            for filename in glob.glob(dir_ + ft_ +'/'+ user + '/*.csv'):
                
                df = pd.read_csv(filename, sep=';|,', engine="python")
                if "loss" in df.columns:
                    df = df.drop("loss", axis=1)
                
                df = df.drop(["Time", "y_hat_Time"], axis=1)
                      
                n = int(len(df.columns)/2)
                y_pred = df.iloc[:, -n:].values
                y_true = df.iloc[:, :n].values
                                
                norm_ = 1
                if "u6h" in ft_ and normalized_:
                    norm_ = 6
                elif "u12h" in ft_ and normalized_:    
                    norm_ = 12
                                                                
                results.append([rmse(y_true/norm_, y_pred/norm_), maape(y_true, y_pred)])
            
            out_filename = out_dir+ft_+'_'+user+".csv"
            if normalized_:
                out_filename = out_dir+"norm_"+ft_+'_'+user+".csv"
            
            pd.DataFrame(results, columns=["RMSE", "MAAPE"]).to_csv(out_filename, sep=';', index=False)


#List of users included in the experiment
user_list = ["sb002", "sb003", "sb006", "sb008"]

dir_UIN = 'PATH TO FOLDER -- LSTM OUTPUT -- UNIFORM INTERVAL NORMALIZATION'
dir_ = 'PATH TO FOLDER -- LSTM OUTPUT -- MINMAX NORMALIZATION + TIME2VEC'

#Set to true to denormalize the values normalized with uniform interval normalization (to correctly calculate RMSE)
#Set to calculate the values normalized with MinMax normalization
normalized_ = True

if normalized_:
    format_ = ["u6h_s3h_time", "u6h_s6h_time", "u12h_s3h_time", "u12h_s6h_time"]
	out_dir = dir_UIN +"/denorm/"
	de_norm(user_list, dir_UIN, out_dir, format_)
	
	dir_UIN = out_dir
    out_dir = 'PATH TO FOLDER -- Desired output for the calc_rmse function -- UNIFORM INTERVAL NORMALIZATION'
	calc_rmse(user_list, dir_UIN, format_, out_dir, normalized_)
	
else:
    out_dir = 'PATH TO FOLDER -- Desired output for the calc_rmse function -- MINMAX NORMALIZATION + TIME2VEC'
	format_ = ["raw_1h", "time2vec"]
	calc_rmse(user_list, dir_, format_, out_dir, normalized_)"""for ft_ in format_:    