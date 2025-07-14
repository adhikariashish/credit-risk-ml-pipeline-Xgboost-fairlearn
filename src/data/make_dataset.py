# File: src/data/make_dataset.py
import pandas as pd
import yaml
from pathlib import Path
from pipeline_config import cfg

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
    df = pd.read_csv(raw_data_path, sep="|", encoding="utf-8", header=None, names=headers)
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


def main():
    # paths
    raw_dir = cfg["data"]["raw_dir"]
    raw_data_path = raw_dir / "2024Q4.csv"
    headers_path =  raw_dir / "crt-file-layout-and-glossary.xlsx"


    interim_dir = cfg["data"]["interim_dir"]
    interim_data_path = interim_dir / "credit_risk_data.csv"

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

    print("✅ Data ingest and clean complete. The cleaned data is saved to:", interim_data_path)



if __name__ == "__main__":
    main()
    print("Make dataset script executed successfully.")     


