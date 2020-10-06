import pandas as pd
import glob

class Merge_user:
    
    def __init__(self):
        None
    
    """
    Input: A folder where the user data is stored as one file for each feature e.g. user1_battery, user1_steps
    Output: One file per user which contains all the features for the user e.g. user1_features
    """
    def merge_user(self, user_list, in_dir, out_dir):
        
        for user_id in user_list:
            
            #if user_id == 'sb002':
                #vars = ['batt', 'call', 'cell', 'screen', 'sms', 'step']
                #for name in vars:
                    #df = pd.read_csv("sb002_"+name+"_data.csv", sep=';')
                    #print(df.head(10))
                    #print(df.loc[df['pretty_time'].duplicated()])
            
            df = pd.concat((pd.read_csv(f, parse_dates=True, sep=';', index_col='pretty_time')
                            for f in glob.glob(in_dir + user_id + '*.csv')), sort=False,axis=1)  
                
                           
            #df = df.drop('user_id', axis = 1)
            df = df.sort_values(by='pretty_time').fillna(0)
            out_path = out_dir + user_id + '_ts.csv' 
            
            #df = df.asfreq(freq='1D').fillna(0)   
            df.to_csv(out_path, sep=';')


    