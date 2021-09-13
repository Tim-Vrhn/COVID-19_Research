import pandas as pd
import numpy as np
from Toolkit import save_file

"""
Replace 0s in specific variables with np.nan
"""

vitals_df = pd.read_csv("Processed_Data/vitals_processed.csv")
features = ['SpO2 (%)', 'Systolic BP (mmHg)', 'Diastolic BP (mmHg)', 'HR (BPM)', 'RR (/min)', 'Temperature (°C)']
print("Removing zeroes for the following features:\n", features)

for feature in features:
    vitals_df[feature].replace(0, np.nan, inplace=True)

""" 
Replace NaN values with 0
"""

features = ['Supplemental Oxygen (L/min)']
for feature in features:
    vitals_df[feature] = vitals_df[feature].fillna(0)

save_file(vitals_df, "Processed_Data/vitals_processed.csv")