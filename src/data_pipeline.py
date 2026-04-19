"""
data_pipeline.py — FailSafe AI Data Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
End-to-end data preparation: loading, cleaning, feature engineering,
class balancing (SMOTE), and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_config, pickle_dump, resolve_path


def load_raw_data(config: dict) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    path = resolve_path(config["dataset"]["data_directory"] + config["dataset"]["file_name"])
    df = pd.read_csv(path)
    print(f"[OK] Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop unnecessary columns and handle missing values."""
    df = df.drop(columns=config["dataset"]["drop_columns"], errors="ignore")
    df = df.dropna()
    print(f"[OK] Cleaned data: {df.shape[0]} rows remaining")
    return df


def rename_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Rename columns to XGBoost-safe names (no brackets)."""
    df = df.rename(columns=config["dataset"]["column_rename"])
    print("[OK] Renamed columns to clean names")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create 3 physics-informed engineered features."""
    df = df.copy()

    # Power [W] = Torque [Nm] x Rotational Speed [rad/s]
    # Convert rpm to rad/s: rpm x 2pi / 60
    df["Power_W"] = df["Torque_Nm"] * df["Rotational_Speed_rpm"] * (2 * np.pi / 60)

    # Temperature difference [K] = Process temp - Air temp
    df["Temp_Diff_K"] = df["Process_Temp_K"] - df["Air_Temp_K"]

    # Overstrain indicator = Tool wear [min] x Torque [Nm]
    df["Overstrain_Indicator"] = df["Tool_Wear_min"] * df["Torque_Nm"]

    print("[OK] Engineered 3 new features: Power_W, Temp_Diff_K, Overstrain_Indicator")
    return df


def encode_type(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Encode the 'Type' column: L=1, M=2, H=3."""
    df = df.copy()
    df["Type"] = df["Type"].map(config["dataset"]["type_encoding"])
    print("[OK] Encoded Type column (L=1, M=2, H=3)")
    return df


def prepare_binary_classification(df: pd.DataFrame, config: dict):
    """
    Prepare data for binary failure prediction (Target: 0 or 1).
    Uses SMOTE to handle the severe class imbalance (96.6% vs 3.4%).
    """
    feature_cols = config["dataset"]["all_features"]
    label_col = config["dataset"]["label"]

    X = df[feature_cols].copy()
    y = df[label_col].copy()

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["dataset"]["test_size"],
        random_state=config["dataset"]["random_state"],
        stratify=y,
    )

    print(f"[DATA] Before SMOTE - Train: {y_train.value_counts().to_dict()}")

    # Apply SMOTE only to training data
    smote = SMOTE(random_state=config["dataset"]["random_state"])
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"[DATA] After SMOTE  - Train: {pd.Series(y_train_res).value_counts().to_dict()}")

    return X_train_res, X_test, y_train_res, y_test


def prepare_failure_type_classification(df: pd.DataFrame, config: dict):
    """
    Prepare data for multi-class failure type prediction.
    """
    feature_cols = config["dataset"]["all_features"]
    ft_label = "Failure_Type"

    X = df[feature_cols].copy()
    y = df[ft_label].copy()

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["dataset"]["test_size"],
        random_state=config["dataset"]["random_state"],
        stratify=y,
    )

    print(f"[DATA] Failure type distribution (train): {y_train.value_counts().to_dict()}")

    # Apply SMOTE for multi-class balancing
    smote = SMOTE(random_state=config["dataset"]["random_state"], k_neighbors=2)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"[DATA] After SMOTE (train): {pd.Series(y_train_res).value_counts().to_dict()}")

    return X_train_res, X_test, y_train_res, y_test


def run_pipeline():
    """Execute the full data pipeline."""
    config = load_config()
    directory = config["train_test"]["directory"]

    # 1. Load & clean
    df = load_raw_data(config)
    df = clean_data(df, config)

    # 2. Rename columns to clean names
    df = rename_columns(df, config)

    # 3. Feature engineering (before encoding, so Type is still string)
    df = engineer_features(df)

    # 4. Encode categorical
    df_encoded = encode_type(df, config)

    # 5. Binary classification data
    print("\n--- Binary Classification (Target) ---")
    X_train, X_test, y_train, y_test = prepare_binary_classification(df_encoded, config)

    pickle_dump(X_train, directory + config["train_test"]["X_train"])
    pickle_dump(X_test,  directory + config["train_test"]["X_test"])
    pickle_dump(y_train, directory + config["train_test"]["y_train"])
    pickle_dump(y_test,  directory + config["train_test"]["y_test"])

    # 6. Failure type classification data
    print("\n--- Failure Type Classification ---")
    X_train_ft, X_test_ft, y_train_ft, y_test_ft = prepare_failure_type_classification(
        df_encoded, config
    )

    pickle_dump(X_train_ft, directory + config["train_test"]["X_train_ft"])
    pickle_dump(X_test_ft,  directory + config["train_test"]["X_test_ft"])
    pickle_dump(y_train_ft, directory + config["train_test"]["y_train_ft"])
    pickle_dump(y_test_ft,  directory + config["train_test"]["y_test_ft"])

    print("\n[OK] Data pipeline complete! All artifacts saved to data/processed/")


if __name__ == "__main__":
    run_pipeline()
