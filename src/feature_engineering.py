import pandas as pd
import numpy as np
import os

def validate_columns(df: pd.DataFrame, required_columns: list) -> None:
    """Validates if all required columns are present in the dataframe."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values safely for feature engineering."""
    # Creating a copy to prevent SettingWithCopyWarning
    df = df.copy()
    # Fill numerical columns with median and categorical with mode or 'Unknown'
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('Unknown')
    return df

def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts month, day, day of week, and weekend flag from order_date."""
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['order_month'] = df['order_date'].dt.month
        df['order_day'] = df['order_date'].dt.day
        df['order_dayofweek'] = df['order_date'].dt.dayofweek
        df['order_is_weekend'] = df['order_dayofweek'].isin([5, 6]).astype(int)
    return df

def calculate_remaining_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates remaining distance using total distance and distance covered percentage."""
    df['remaining_distance_km'] = df['distance_km'] * (1 - (df['distance_covered_pct'] / 100.0))
    return df

def calculate_vendor_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates vendor reliability score combining supplier score and historical on-time performance."""
    df['vendor_reliability_score'] = (df['supplier_score'] * df['historical_ontime_pct']) / 100.0
    return df

def calculate_warehouse_stress(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates warehouse stress score using capacity and mapping of inventory level."""
    inv_map = {'High': 1.5, 'Medium': 1.0, 'Low': 0.5, 'Unknown': 1.0}
    if df['inventory_level'].dtype == 'object':
        inv_multiplier = df['inventory_level'].map(inv_map).fillna(1.0)
    else:
        inv_multiplier = df['inventory_level']
    df['warehouse_stress_score'] = df['warehouse_capacity_pct'] * inv_multiplier
    return df

def calculate_vehicle_health(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates vehicle health score comparing driver experience vs vehicle age."""
    df['vehicle_health_score'] = df['driver_experience_years'] / (df['vehicle_age_years'] + 1)
    return df

def calculate_route_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates route complexity incorporating risk score and total distance."""
    df['route_complexity_score'] = df['route_risk_score'] * (df['distance_km'] / 100.0)
    return df

def calculate_delivery_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates delivery efficiency by assessing distance covered per planned ETA day."""
    # Safely handle division by zero where planned_eta_days is 0
    df['delivery_efficiency'] = np.where(df['planned_eta_days'] > 0,
                                         df['distance_km'] / df['planned_eta_days'],
                                         df['distance_km'])
    return df

def calculate_fuel_cost_index(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates an overall fuel cost index for the trip."""
    df['fuel_cost_index'] = df['fuel_price_index'] * df['distance_km']
    return df

def calculate_weather_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates weather risk by numerically mapping weather conditions."""
    weather_map = {'Clear': 0, 'Rain': 1, 'Snow': 2, 'Storm': 3, 'Fog': 1.5, 'Unknown': 1}
    if 'weather' in df.columns and df['weather'].dtype == 'object':
        df['weather_risk'] = df['weather'].map(weather_map).fillna(1)
    else:
        df['weather_risk'] = df['weather']
    return df

def calculate_traffic_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates traffic risk scaling traffic index by distance."""
    df['traffic_risk'] = df['traffic_index'] * (df['distance_km'] / 100.0)
    return df

def run_feature_engineering_pipeline(input_path: str, output_path: str) -> None:
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Validate required columns
    required_cols = [
        'distance_km', 'distance_covered_pct', 'supplier_score', 'historical_ontime_pct',
        'warehouse_capacity_pct', 'inventory_level', 'vehicle_age_years', 
        'driver_experience_years', 'route_risk_score', 'planned_eta_days',
        'fuel_price_index', 'weather', 'traffic_index'
    ]
    validate_columns(df, required_cols)
    
    # Handle missing values safely
    df = handle_missing_values(df)
    
    # Track columns before to identify new features
    initial_columns = set(df.columns)
    
    # Apply feature engineering steps
    print("Applying feature engineering...")
    df = extract_datetime_features(df)
    df = calculate_remaining_distance(df)
    df = calculate_vendor_reliability(df)
    df = calculate_warehouse_stress(df)
    df = calculate_vehicle_health(df)
    df = calculate_route_complexity(df)
    df = calculate_delivery_efficiency(df)
    df = calculate_fuel_cost_index(df)
    df = calculate_weather_risk(df)
    df = calculate_traffic_risk(df)
    
    # Identify and print new features
    new_columns = set(df.columns) - initial_columns
    print("\nNewly created features:")
    for col in sorted(list(new_columns)):
        print(f" - {col}")
        
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the processed dataset
    df.to_csv(output_path, index=False)
    print(f"\nFeature engineered data saved successfully to {output_path}")

# Run the pipeline
input_file = "../data/processed/cleaned_data.csv"
output_file = "../data/processed/feature_engineered_data.csv"
run_feature_engineering_pipeline(input_file, output_file)

