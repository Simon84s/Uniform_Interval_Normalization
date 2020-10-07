import pandas as pd

class Split_col:
    
    def __init__(self):
        None
    
    """
    First select a column to split, then input values to split on
    The data frame is then split into new data frames one for each value 
    E.g., if column that stores incoming and outgoing calls are selected 
    two new data frames are created one with incoming and one with outgoing calls

    Also bins the columns into segments, the lengths of the segments are determined by bin_var
    """
    def split_col(self, feature, in_dir, out_dir, user_list, select_col, split_values):
        
        #Modify so it can load the required file        
        file_path = '_' + feature + '_data.csv'
                
        for user in user_list:
            
            try:
                df = pd.read_csv(in_dir + user + file_path, sep=';')
            except:
                continue
            
            split_files = []
            file_paths = []
            
            for value in split_values:
                
                df_new = df[df[select_col] == value]
                if feature =='call':
                    df_new = df_new.rename(columns={'duration':'duration ' + value}).drop(['direction'], axis=1)
                                        
                elif feature == 'sms':
                    df_new = df_new.rename(columns={'body_length':'body_length ' + value}).drop(['direction'], axis=1)
                
                elif feature == 'screen':
                    df_new = df_new.rename(columns={'status':'status ' + value})    
                     
                split_files.append(df_new)
                                
                file_paths.append(out_dir +user + '_' + value + file_path)
                        
            for i in range(len(split_files)):
                split_files[i].to_csv(file_paths[i], sep=';', index = False)              
        
