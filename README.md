# Uniform_Interval_Normalization
Codebase for my master thesis in Data Science

* extract_users_to_files.py: The MoodMapper data sets had one file per feature, which contained data for all users. This script extracts and stores the data in one file per user instead
* Split_col.py: splits a feature into two features based on some value (e.g., calls -> incoming/outgoing)
* Binning.py: bins a data frame into a time series with equal steps between each data point, fills empty data points with zero
* norm.py: Returns one file per feature with the normalization factor for the feature (still needs to be manually entered in Dist_transform.py).
* Dist_transform.py: performs the uniform interval normalizaton.
* Merge_user: Merges the files into one time series per user. 
