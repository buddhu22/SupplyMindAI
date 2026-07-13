import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

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

def verify_data(df: pd.DataFrame, target_col: str) -> None:
    """Verify dataset shape and presence of target column."""
    logger.info(f"Dataset shape: {df.shape}")
    
    if df.empty:
        error_msg = "Dataset is empty."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if target_col not in df.columns:
        error_msg = f"Target column '{target_col}' not found in dataset."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    logger.info(f"Target column '{target_col}' verified successfully.")

def split_features_target(df: pd.DataFrame, target_col: str):
    """Separate features (X) and target (y)."""
    logger.info("Separating features (X) and target (y).")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def save_data(data: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame/Series to a CSV file."""
    logger.info(f"Saving data to {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        data.to_csv(filepath, index=False)
        logger.info(f"Successfully saved {filepath}.")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}")
        raise

def main():
    # File paths
    input_filepath = "data/processed/encoded_data.csv"
    output_dir = "data/processed"
    target_column = "delay"
    
    X_train_path = os.path.join(output_dir, "X_train.csv")
    X_test_path = os.path.join(output_dir, "X_test.csv")
    y_train_path = os.path.join(output_dir, "y_train.csv")
    y_test_path = os.path.join(output_dir, "y_test.csv")

    try:
        # 1. Load Data
        df = load_data(input_filepath)

        # 2. Verify Data
        verify_data(df, target_column)
        
        # Print dataset shape as required
        print(f"Dataset shape: {df.shape}")

        # 3. Separate Features and Target
        X, y = split_features_target(df, target_column)

        # 4. Train-Test Split
        logger.info("Performing train-test split (test_size=0.20, random_state=42, stratify=y)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.20, 
            random_state=42, 
            stratify=y
        )
        
        # 5. Print shapes
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        logger.info(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        # 6. Save split datasets
        save_data(X_train, X_train_path)
        save_data(X_test, X_test_path)
        save_data(y_train, y_train_path)
        save_data(y_test, y_test_path)
        
        logger.info("Train-Test Split stage completed successfully.")

    except Exception as e:
        logger.error(f"Train-Test Split pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
