import pandas as pd
import numpy as np
import os

print("Starting data preprocessing...")
input_path = 'data/SupplyChainSentinel_Enterprise_100000_Fixed_2(Sheet1).csv'
output_path = 'data/SupplyChain_Cleaned.csv'

# Load data
df = pd.read_csv(input_path)

# 1. Handle Missing Values
print("Handling missing values...")
# If delay is 'No', delay_reason is usually NaN. Fill with 'No Delay'
if 'delay_reason' in df.columns:
    df['delay_reason'] = df['delay_reason'].fillna('No Delay')

# For numerical columns, fill with median
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# For categorical columns, fill with mode
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# 2. Date Formatting
print("Formatting dates...")
if 'order_date' in df.columns:
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
if 'gps_last_updated' in df.columns:
    df['gps_last_updated'] = pd.to_datetime(df['gps_last_updated'], errors='coerce')

# 3. Save cleaned data
print(f"Saving cleaned dataset to {output_path}...")
df.to_csv(output_path, index=False)
print("Data preprocessing completed successfully!")
