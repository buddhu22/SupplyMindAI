import os
import sys
import logging
import pickle
import pandas as pd
from xgboost import XGBClassifier

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

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Initialize and train the XGBoost model."""
    logger.info("Initializing XGBClassifier...")
    model = XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )
    
    logger.info("Training started.")
    print("Training started...")
    
    try:
        model.fit(X_train, y_train)
        logger.info("Training completed.")
        print("Training completed.")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def save_model(model: XGBClassifier, filepath: str) -> None:
    """Save the trained model as a pickle file."""
    logger.info(f"Saving model to {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Successfully saved model to {filepath}.")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main():
    # File paths
    data_dir = "data/processed"
    X_train_path = os.path.join(data_dir, "X_train.csv")
    X_test_path = os.path.join(data_dir, "X_test.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")
    
    model_path = "models/xgboost_model.pkl"

    try:
        # 1. Load Data
        X_train = load_data(X_train_path)
        X_test = load_data(X_test_path)
        y_train = load_data(y_train_path).squeeze()
        y_test = load_data(y_test_path).squeeze()
        
        # Map 'No' to 0 and 'Yes' to 1 (XGBoost requires int/bool for target)
        y_train = y_train.replace({'No': 0, 'Yes': 1}).astype(int)
        y_test = y_test.replace({'No': 0, 'Yes': 1}).astype(int)
        
        # 2. Print shapes and number of features
        print("-" * 30)
        print("DATASET SHAPES")
        print("-" * 30)
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"Number of features: {X_train.shape[1]}")
        print("-" * 30)

        # 3. Train the model
        model = train_model(X_train, y_train)

        # 4. Generate predictions and probabilities
        logger.info("Generating predictions on the test set.")
        predictions = model.predict(X_test)
        prediction_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1
        
        # 5. Print Sample predictions
        print("-" * 30)
        print("SAMPLE PREDICTIONS")
        print("-" * 30)
        print(f"Sample Predictions (first 5): {predictions[:5]}")
        print(f"Sample Probabilities (first 5): {prediction_probs[:5]}")
        print("-" * 30)

        # 6. Save model
        save_model(model, model_path)
        
        logger.info("Model Training stage completed successfully.")

    except Exception as e:
        logger.error(f"Model Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
