import pandas as pd
import numpy as np

def audit_dataset(csv_path: str):
    print("=" * 60)
    print("BATTERY DATASET AUDIT & CONFIG GENERATOR")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"\n1. Basic Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns found: {list(df.columns)}")

    # 1. Unique Groups (Batteries)
    print("\n2. Unique Battery Identifiers:")
    battery_counts = df['battery_id'].value_counts()
    print(battery_counts)
    batteries = list(battery_counts.index)

    # 2. Target & Feature Correlation (Leakage Detection)
    if 'rul' in df.columns:
        print("\n3. Feature Correlation with Target (RUL):")
        correlations = df.corr(numeric_only=True)['rul'].sort_values(ascending=False)
        print(correlations)
        
        # High correlation warning (Potential Leakage)
        suspicious = [col for col in correlations.index if col != 'rul' and abs(correlations[col]) > 0.85]
        print(f"\n[ALERT] Highly correlated columns (>0.85): {suspicious}")
        print("Reason to drop: These directly reveal ground truth or are derived targets.")

    # 3. Physical Signal Ranges
    print("\n4. Statistical Bounds for Sensor Readings:")
    stats = df[['voltage', 'temperature']].describe().T[['min', 'mean', 'max']]
    print(stats)

    # 4. Recommended Config Output
    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIG PARAMS (Copy these to config.yaml):")
    print("=" * 60)
    print(f"train_batteries: {batteries[:-1]}")
    print(f"test_batteries: {[batteries[-1]]}")
    print(f"leakage_columns: {suspicious}")
    print(f"validation_ranges:")
    print(f"  voltage_min: {np.floor(df['voltage'].min() - 0.5)}")
    print(f"  voltage_max: {np.ceil(df['voltage'].max() + 0.5)}")
    print(f"  temp_min: {np.floor(df['temperature'].min() - 5)}")
    print(f"  temp_max: {np.ceil(df['temperature'].max() + 5)}")
    print("=" * 60)

if __name__ == "__main__":
    audit_dataset("data/raw/battery_cycle_level_dataset_CLEAN_FINAL.csv")