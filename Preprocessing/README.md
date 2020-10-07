# Files used to prepare the data for the LSTM network

* extract_users_to_files.py: The MoodMapper data sets had one file per feature, which contained data for all users. This script extracts and stores the data in one file per user instead
* Split_col.py: splits a feature into two features based on some value (e.g., calls -> incoming/outgoing)
* Binning.py: transforms a series of data point with timestamps, into a equidistant time series.
* norm.py: calculates and returns the normalization factor. The ouput is one file per feature and the factor is in the column titled "Max".
* Dist_transform.py: performs the uniform interval normalizaton.
* Merge_user: Merges the files into one time series per user. 

*main.py: The main file used during the thesis project, and can be used as an example on how to call the scripts.
