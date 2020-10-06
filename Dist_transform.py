import numpy as np
import math
import glob
import pandas as pd

class Dist_transform:
    
    def __init__(self):
        None
    
    #Return dictionary of the normalization value for the uniform distribution
    #Currently filled in manually, the norm value for each feature is found in the norm folder (the column named Max) 
    def get_norm_value(self):
        return {"body_length incoming": 1593,    
                "body_length outgoing": 1431,  
                "cell_tower_hash": 68,                
                "duration incoming": 11401,
                "duration outgoing": 9402,   
                "status off": 201,
                "status on": 236,
                "steps": 2084,                  
                }
            
    def transform(self, user_list, in_path, out_path, distribution, segment_length, uniform_length, step):
        
        if distribution == "uniform":
            for user in user_list:
                for filename in glob.glob(in_path + user + '*.csv'):
                    
                    df = pd.read_csv(filename, sep=';')  
                                        
                    #Keep every n:th date, where n is equal to the step between each uniform distribution
                    pt = df['pretty_time'][::step].reset_index(drop=True)
                    df = df.drop('pretty_time', axis=1)
                               
                    for col_name in df.columns:

                        uni_dist = self.uniform(df.to_numpy().flatten(), 
                                                segment_length, 
                                                uniform_length, 
                                                step, 
                                                self.get_norm_value()[col_name])

                        df = pd.concat([pt, pd.DataFrame(uni_dist).rename(columns={0: col_name})], axis=1) 
                        df = df.dropna()
                    
                        df.to_csv(out_path + user + "_" + col_name + "_uniform.csv", sep=';', index=False) 
                
    """
    time_series: input time series
    segment_length: e.g., 24h, length of time period for the time_series.
    uniform_length: number of time steps to sum up as basis for the uniform distribution.
    norm_value: value to divide the sum with
    """ 
    def uniform(self, time_series, segment_length, uniform_length, step, norm_value):
        
        ts = []
        i = 0
            
        pad_width = math.floor(uniform_length/2)
        
        #Add a number of zeroes equal to the length of a segment-1 in order to not go out of bounds at the last values of the list
        #Zeroes are also added in the beginning of the list so values both before and after the current hour will be included in the interval
        time_series = np.pad(time_series, (pad_width, pad_width), 'constant')
            
        while(i+segment_length) < len(time_series):
            #segment = time_series[i:i + segment_length]
                        
            #Iterate over the hours in the segment, and calculate a uniform distribution over the length of the uniform_length
            
            #i iterates over the full list list[i]
            #j iterates over the segments, list[i][j]
            #step determines with how much j increases each iteration
            for j in range(i, i+segment_length, step):
             
                #sum values starting from current hour and until hours equal to uniform length have been added             
                ts.append(np.sum(time_series[j:j+uniform_length]) / norm_value)
                                        
            #Move one day forward
            i += 24
                
        return np.array(ts)   