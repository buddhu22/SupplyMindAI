import pandas as pd
import numpy as np
import os

input_path = '../data/SupplyChainSentinel_Enterprise_100000_Fixed_2(Sheet1).csv'
output_path = '../data/SupplyChain_Cleaned.csv'

print("Loading dataset...")
df = pd.read_csv(input_path)
df.head()

# Delay Reason
if 'delay_reason' in df.columns:
    df['delay_reason'] = df['delay_reason'].fillna('No Delay')

# Numerical columns
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Categorical columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled.")

if 'order_date' in df.columns:
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
if 'gps_last_updated' in df.columns:
    df['gps_last_updated'] = pd.to_datetime(df['gps_last_updated'], errors='coerce')

print("Dates formatted.")

print(f"Saving cleaned dataset to {output_path}...")
df.to_csv(output_path, index=False)
print("Data preprocessing completed successfully!")

