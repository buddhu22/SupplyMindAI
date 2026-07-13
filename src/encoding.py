import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from the specified filepath."""
    logger.info(f"Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def validate_columns(df: pd.DataFrame, expected_columns: list) -> None:
    """Validate that all expected columns exist in the DataFrame."""
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"Missing required columns in dataset: {missing_columns}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info("All required columns are present.")

def handle_missing_values(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Safely handle missing values for categorical features."""
    logger.info("Handling missing values.")
    df_clean = df.copy()
    
    # Separate categorical columns to fillna with 'Unknown'
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
        
    for col in cat_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna('Unknown')
            logger.info(f"Filled missing values in categorical column '{col}' with 'Unknown'")
            
    # For target column, drop rows with missing target
    if df_clean[target_col].isnull().any():
        df_clean = df_clean.dropna(subset=[target_col]).reset_index(drop=True)
        logger.info(f"Dropped rows with missing target variable '{target_col}'")
        
    return df_clean

def encode_features(df: pd.DataFrame, target_col: str, ordinal_cols: list, nominal_cols: list) -> pd.DataFrame:
    """Apply Label Encoding and One-Hot Encoding to categorical features."""
    logger.info("Starting feature encoding.")
    df_encoded = df.copy()
    
    # 1. Separate Target
    target_series = df_encoded.pop(target_col)
    logger.info(f"Target column '{target_col}' separated.")
    
    # Automatically identify categorical columns
    auto_cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    logger.info(f"Automatically identified categorical columns: {auto_cat_cols}")
    
    # Add any unidentified categorical columns to ordinal_cols for Label Encoding by default to save memory
    for col in auto_cat_cols:
        if col not in ordinal_cols and col not in nominal_cols:
            logger.warning(f"Column '{col}' is categorical but not specified as ordinal or nominal. Defaulting to Label Encoding.")
            ordinal_cols.append(col)
    
    # 2. Label Encoding for Ordinal features
    label_encoder = LabelEncoder()
    encoded_ordinal_count = 0
    for col in ordinal_cols:
        if col in df_encoded.columns:
            logger.info(f"Applying Label Encoding to '{col}'")
            # Convert to string to avoid mixed type errors during encoding
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col].astype(str))
            encoded_ordinal_count += 1
            
    # 3. One-Hot Encoding for Nominal features
    encoded_nominal_cols = [col for col in nominal_cols if col in df_encoded.columns]
    logger.info(f"Applying One-Hot Encoding to nominal columns: {encoded_nominal_cols}")
    
    # Perform pd.get_dummies
    if encoded_nominal_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=encoded_nominal_cols, drop_first=True)
        # Convert boolean columns from get_dummies to integers (0/1) for better compatibility
        bool_cols = df_encoded.select_dtypes(include=['bool']).columns
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    
    # 4. Re-attach Target
    df_encoded[target_col] = target_series
    logger.info(f"Target column '{target_col}' re-attached.")
    
    # 5. Validation and Reporting
    remaining_cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in remaining_cat_cols:
        # Target might be categorical natively, we ignore it for this check
        remaining_cat_cols.remove(target_col)
         
    total_encoded = encoded_ordinal_count + len(encoded_nominal_cols)
    
    print("-" * 30)
    print("ENCODING SUMMARY")
    print("-" * 30)
    print(f"Dataset shape after encoding: {df_encoded.shape}")
    print(f"Number of originally encoded columns: {total_encoded}")
    print(f"Remaining categorical columns: {remaining_cat_cols}")
    print("-" * 30)
    
    if not remaining_cat_cols:
        logger.info("All categorical features successfully encoded.")
    else:
        logger.warning(f"Some categorical columns were not encoded: {remaining_cat_cols}")

    return df_encoded

def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save the encoded dataset to a CSV file."""
    logger.info(f"Saving encoded dataset to {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        df.to_csv(filepath, index=False)
        logger.info("Successfully saved encoded data.")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}")
        raise

def main():
    # File paths
    input_filepath = "data/processed/feature_engineered_data.csv"
    output_filepath = "data/processed/encoded_data.csv"
    
    # Configuration
    target_column = "delay"
    ordinal_features = ["inventory_level", "weather_risk", "traffic_risk"]
    nominal_features = [
        "vendor", 
        "material", 
        "vehicle_type", 
        "project_type", 
        "weather", 
        "origin_state", 
        "destination_state"
    ]
    
    required_columns = [target_column] + ordinal_features + nominal_features
    
    try:
        # Load dataset
        df = load_data(input_filepath)
        
        # Validate required columns
        validate_columns(df, required_columns)
        
        # Handle missing values safely
        df_clean = handle_missing_values(df, target_column)
        
        # Encode features
        df_encoded = encode_features(
            df=df_clean, 
            target_col=target_column, 
            ordinal_cols=ordinal_features, 
            nominal_cols=nominal_features
        )
        
        # Save output
        save_data(df_encoded, output_filepath)
        
    except Exception as e:
        logger.error(f"Encoding pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
