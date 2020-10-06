import pandas as pd
import glob

class Extract_users: 

    def __init__(self):
        None 
    
    """ Input: Name of the variable corresponding to a file  
        
        Output: .csv files with the variable data for one user 
    """
    def extract_users_to_files(self, feature, split_dir, not_split_dir):
        file_path = feature + '_data2.csv'
        df = pd.read_csv(file_path, sep=';')
        csv_path = '_' + feature + '_data.csv'
        
        user_list = df.user_id.unique()
        split_list = ['call', 'screen', 'sms']    
        
        for user_id in user_list:
                        
            df_extracted = df.loc[df['user_id'] == user_id]
            
            #remove duplicates values in pretty time
            df_extracted = df_extracted.loc[~df_extracted['pretty_time'].duplicated(keep='first')]
            
            df_extracted = df_extracted.drop(['user_id'], axis=1)
            
            #If csv does not need further processing
            if feature not in split_list:
                out_path = not_split_dir + user_id + csv_path
            #If csv has columns that must be split
            else:
                out_path = split_dir +user_id + csv_path
            
            df_extracted.to_csv(out_path, sep=';', index=False)
                        
            # or 
            #df.loc[df['user_id'] == user_id].to_csv(user_id+csv_path, sep=',', index=False)
        return user_list
    
    