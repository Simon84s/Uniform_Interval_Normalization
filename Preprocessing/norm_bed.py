import pandas as pd
import numpy as np
import math

def get_norm(time_series, segment_length, uniform_length, step):
        
        ts = []
        i = 0
               
        pad_width = math.floor(uniform_length/2)
        
        #Add a number of zeroes equal to the length of a segment-1 in order to not go out of bounds at the last values of the list
        #Zeroes are also added in the beginning of the list so values both before and after the current hour will be included in the interval
        time_series = np.pad(time_series, (pad_width, pad_width), 'constant')
        
        #skip the first hours equal to half the uniform length sinc ethere are not enough data points to calculate a distribution
        #i = math.floor(uniform_length/2) 
        
        while(i+segment_length) < len(time_series):
            #segment = time_series[i:i + segment_length]
                        
            #Iterate over the hours in the segment, and calculate a uniform distribution over the length of the uniform_length
            
            #i iterates over the full list list[i]
            #j iterates over the segments, list[i][j]
            #step determines with how much j increases each iteration
            for j in range(i, i+segment_length, step):
             
                #sum values starting from current hour and until hours equal to uniform length have been added             
                ts.append(np.sum(time_series[j:j+uniform_length]))
                                        
            #Move one day forward (if segment_length == 24h)
            i += segment_length
        
        
        return np.array(ts).max()


"""
time_series == univariate time series
Segment_length = number of time steps to repeat a cycle of intervals e.g., if 6 intervals are created per day segment length is 24h  

"""
def uniform(time_series, segment_length, uniform_length, step, norm_value):
        
        ts = []
        i = 0
               
        pad_width = math.floor(uniform_length/2)
        
        #Add a number of zeroes equal to the length of a segment-1 in order to not go out of bounds at the last values of the list
        #Zeroes are also added in the beginning of the list so values both before and after the current hour will be included in the interval
        time_series = np.pad(time_series, (pad_width, pad_width), 'constant')
        
        #skip the first hours equal to half the uniform length since there are not enough data points to calculate a distribution
        #i = math.floor(uniform_length/2) 
        
        while(i+segment_length) < len(time_series):
            #segment = time_series[i:i + segment_length]
                        
            #Iterate over the hours in the segment, and calculate a uniform distribution over the length of the uniform_length
            
            #i iterates over the full list list[i]
            #j iterates over the segments, list[i][j]
            #step determines with how much j increases each iteration
            for j in range(i, i+segment_length, step):
             
                #sum values starting from current hour and until hours equal to uniform length have been added             
                ts.append(np.sum(time_series[j:j+uniform_length]) / norm_value)
                                        
            #Move one day forward (if segment_length == 24h)
            i += segment_length
        
        
        return np.array(ts)


df = pd.read_csv("df_bed_1m_2w_10.csv", sep=';')
feature_list = ['Scale 1', 'Scale 2', 'Scale 3', 'Scale 4']
"""
# Segment_length represents the length of a 'full cycle'
# For MoodMapper one day (24h) is considered a 'full cycle'
# For SafeBase one hour is considered a 'full cycle' 
"""
segment_length = 60
uniform_length = 60
step = 20

pt = df['pretty_time'][::step].reset_index(drop=True)
norm_ls = []
for col in df.columns:
    if "pretty_time" not in col:
        norm_ = get_norm(df[col], segment_length, uniform_length, step)
        interval_ = uniform(df[col], segment_length, uniform_length, step, norm_)
        norm_ls.append(norm_)
        pt = pd.concat([pt, pd.DataFrame(interval_).rename(columns={0: col})], axis=1) 
        pt = pt.dropna()

pt = pd.concat([pt, pd.DataFrame(norm_ls)], axis=1) 
pt.to_csv("bed_u60m_s20m.csv", sep=';', index=False)