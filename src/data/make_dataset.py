# File: src/data/make_dataset.py

import pandas as pd
import yaml
from pathlib import Path
import argparse
import re
import os

def load_headers(headers_path: Path) -> list[str]:
    """
    importing the header file first from the Excel file.
    """
    df_headers = pd.read_excel(headers_path, sheet_name="Combined Glossary", usecols=["Field Name"])
    headers = (df_headers["Field Name"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .str.replace(r"[ \(\)/\-]+", "_", regex=True)
                        .str.replace(r"[^a-z0-9_]", "", regex=True)
                        .tolist())
    return headers

def load_data(raw_data_path: Path, headers: list[str]) -> pd.DataFrame:
    """
    Load the raw data and return a DataFrame.
    """
    # Defining a dtype map for those two columns which is giving dtypewarning:
    col_101 = headers[101]
    col_104 = headers[104]

    dtype_map = {
        col_101: "string",   
        col_104: "string",     
    }
    
    df = pd.read_csv(raw_data_path, sep="|", encoding="utf-8", header=None, names=headers, dtype=dtype_map, low_memory=False)

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data and return a DataFrame.
    """
    # add target column
    df["default_flag"] = (df["current_loan_delinquency_status"].astype(int) > 0
            ).astype(int)

    # removing the duplicate rows
    df = df.drop_duplicates()

    # dropping the columns with null values
    df = df.dropna(axis=1, how="all")

    # dropping the rows with null values
    df = df.dropna(axis=0, how="all")

    return df

def save_data(df: pd.DataFrame, interim_data_path: Path) -> None:
    """
    Save the cleaned data to the interimediate directory.
    """
    interim_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(interim_data_path, index=False)

def parse_args():
    p = argparse.ArgumentParser(description="Clean raw loan data by quarter(s)")
    p.add_argument(
        "--quarters", "--quarter", "-q",
        dest="quarters",
        required=True,
        help="Quarter tag(s), e.g. '2024Q4' or '2024Q1,2024Q2'"
    )
    p.add_argument("--config", "-c", default=None, help="Path to config.yaml (optional)")
    return p.parse_args()


def main():

    args = parse_args()

    # If provided, point pipeline_config at a custom YAML BEFORE importing cfg
    if args.config:
        os.environ["CR_CONFIG_PATH"] = args.config

    from pipeline_config import cfg

    # Normalize quarters into a list
    quarters = [q.strip() for q in args.quarters.split(",") if q.strip()]

    for quarter in quarters:
        if not re.fullmatch(r"\d{4}Q[1-4]", quarter):
            raise ValueError(f"Invalid quarter: {quarter} (expected YYYYQn, e.g. 2024Q4)")


        # Validate the quarter format: four digits + ‘Q’ + 1–4
        if not re.fullmatch(r"\d{4}Q[1-4]", quarter):
            raise ValueError(
                f"\n\n❌ Invalid quarter format: '{quarter}'.\n"
                "✅ Expected format is YYYYQn, where n is 1, 2, 3, or 4, e.g. 2024Q4.\n"
            )

        # Build the raw‐data path
        raw_dir = Path(cfg["data"]["raw_dir"])
        raw_data_path = raw_dir / cfg["templates"]["raw"].format(quarter=quarter)
        headers_path = raw_dir / "crt-file-layout-and-glossary.xlsx"

        # Check that file exists
        if not raw_data_path.exists():
            raise FileNotFoundError(
                f"\n\n🚨 Raw data file not found: {raw_data_path}\n"
                f"🔍 Looking for: {cfg['templates']['raw'].format(quarter=quarter)}\n"
            )

        # Build the interim output path
        interim_dir  = Path(cfg["data"]["interim_dir"])
        interim_dir.mkdir(parents=True, exist_ok=True)
        interim_data_path = interim_dir / cfg["templates"]["interim"].format(quarter=quarter)

        print( "▶️  Loading the raw data...")
        headers = load_headers(headers_path)

        df_raw = load_data(raw_data_path, headers)

        print("✅  Raw data loaded successfully.")

        # Process the data
        print("▶️  Cleaning the raw data...")
        df_cleaned = clean_data(df_raw)

        # saving the cleaned data
        print("▶️  Saving the cleaned data...")
        save_data(df_cleaned, interim_data_path)

        print(f"✅ Data ingest and clean complete. The cleaned data is saved to: data/interim/credit_risk_raw_{quarter}.csv")



if __name__ == "__main__":

    main()

    print("🍾 Make dataset script executed successfully.")     


