from Extract_users import Extract_users
from Split_col import Split_col
from Merge_user import Merge_user
from Binning import Bin
import norm
import pandas as pd
from Dist_transform import Dist_transform

#Show as many columns that fits the console
pd.options.display.width = 0

#Names of the features to include 
feature_list = ['call', 'screen', 'cell', 'sms', 'step']

"""split values: 
    Calls  | direction -> incoming/outgoing/missed
    Screen | on -> True/False
    Sms    | is_outbound -> True/False
"""
user_list = []
#Used to bin/sample values
bin_var = '1H'
#Only  uniform distribution available
distribution = "uniform"
segment_length = 24
#The length of one interval
uniform_length = 12
#The number of steps between the start of two intervals
step = 6

for f in feature_list:
    user_list = Extract_users().extract_users_to_files(f,split_dir='/split/',
                                           not_split_dir='/user_data/')
        
    if f == 'call': 
        Split_col().split_col(f, user_list, 
                              in_dir='/split/', 
                              out_dir='/user_data/', 
                              'direction', ['incoming', 'outgoing', 'missed'])
    
    elif f == 'sms':
        Split_col().split_col(f, user_list,
                              in_dir='/split/', 
                              out_dir='/user_data/',                               
                              'direction', ['incoming', 'outgoing'])
    
    elif f == 'screen': 
        Split_col().split_col(f, user_list, 
                              in_dir='/split/', 
                              out_dir='/user_data/', 
                              'status', ['on', 'off'])
        

user_list = ["List of users"]

Bin().bin(user_list, bin_var, in_dir='/user_data/', 
          out_dir='/binned/')

norm.get_norm(user_list, uniform_length, 
              in_dir='/binned/', 
              tmp_dir='/binned_1/',
              out_dir='/Norm/')

Dist_transform().transform(user_list, in_dir='/binned/', 
                           out_dir='/uniform/', 
                           distribution, segment_length, uniform_length, step)

Merge_user().merge_user(user_list, in_dir='/uniform/',
                         out_dir='/user time series/')
