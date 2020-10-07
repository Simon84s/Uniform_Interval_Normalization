import pandas as pd
import glob


class Bin:
    
    def __init__(self):
        None
        
    #list of users
    #bin_var, parameter for the frequency of pandas grouper function  
    #in_path e.g., 'root/user_data/'
    #out_path e.g., root/binned/
    
    def bin(self, user_list, start_date='2017-08-01 00:00:00', bin_var, in_path, out_path):
                          
        start_date = pd.to_datetime(start_date)
        #list with the name of feature to aggregate by sum function
        feature_sum = ["""feature_name_1, feature_name_2, etc"""] 
        
        #list with the name of feature to aggregate by mean function
        feature_avg = ["""feature_name_1, feature_name_2, etc"""] 
        
        #list with the name of features to aggregate by count function
        feature_count = ["""feature_name_1, feature_name_2, etc"""]
                
        for user in user_list:
            for filename in glob.glob(in_path + user + '*.csv'):
                df = pd.read_csv(filename, sep=';')
               
                #Convert index to date time format, in order to perform date operations
                df['pretty_time'] = pd.to_datetime(df['pretty_time'])
                
                bin_columns = []
                bin_columns.append(pd.Grouper(key='pretty_time', freq=bin_var))
                df.pretty_time = pd.to_datetime(df.pretty_time)
                df = df[df['pretty_time'] > start_date]
                df = pd.concat([pd.DataFrame([[start_date]], columns=['pretty_time']),df]) 
                                
                for col_name in df.columns:
                    
                    if len(df[col_name].index) <= 1:
                        continue 
                    
                    if col_name in feature_sum:  
                        df = df.groupby(bin_columns).sum().reset_index()
                                                                                           
                    elif col_name in feature_avg:
                        df = df.groupby(bin_columns).mean().reset_index()
                    
                    elif col_name in feature_count:
                        df = df.groupby(bin_columns).count().reset_index()

                    else:
                        #if unknown column do nothing and continue to next column
                        continue
                    
                    df.to_csv(out_path+user+'_'+col_name+'_data.csv', sep=';', index=False)
            
            
