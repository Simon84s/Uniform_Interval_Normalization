import numpy as np
import pandas as pd
import glob

"""
Returns the uniform interval normalization factor
The output is one file per feature and the normalization factor as a column titled Max 
"""
def get_norm(user_list, uniform_length, in_dir, tmp_dir, out_dir):
        
    #list with the name of feature to aggregate by sum function
    feature_sum = ["""feature_name_1, feature_name_2, etc"""] 
        
    #list with the name of feature to aggregate by mean function
    feature_avg = ["""feature_name_1, feature_name_2, etc"""] 
        
    #list with the name of features to aggregate by count function
    feature_count = ["""feature_name_1, feature_name_2, etc"""]
    
    feature_list = feature_sum
    freq_ = str(uniform_length)+'H'
                
    for user in user_list:
        for filename in glob.glob(in_dir + user + '*.csv'):
            df = pd.read_csv(filename, sep=';')
               
            #Convert index to date time format, in order to perform date operations
            df['pretty_time'] = pd.to_datetime(df['pretty_time'])
                
            bin_columns = []
            bin_columns.append(pd.Grouper(key='pretty_time', freq=freq_))
                                        
            for col_name in df.columns:
                    
                if col_name in feature_sum:  
                    df = df.groupby(bin_columns).sum().reset_index()
                                                
                elif col_name in feature_avg:
                    df = df.groupby(bin_columns).mean().reset_index()
                                                
                elif col_name in feature_count:
                    df = df.groupby(bin_columns).count().reset_index()
                                                
                else:
                    #if unknown column do nothing and continue to next column
                    continue
                    
                df.to_csv(tmp_dir + user + '_' + col_name + '_data.csv', sep=';', index=False)

    for feature in feature_list:
                    
        df = pd.concat((pd.read_csv(f, parse_dates=True, sep=';', index_col='pretty_time')
            for f in glob.glob(tmp_dir + '*' + feature + '*.csv')), sort=False,axis=1)  
                    
        df = df.fillna(0)                        
        df['Max'] = df.values.max()
            
        out_path = out_dir + feature + '_norm.csv' 
               
        df.to_csv(out_path, sep=';')

