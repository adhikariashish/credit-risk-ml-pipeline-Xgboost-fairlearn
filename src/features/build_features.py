# File: src/features/build_features.py
# importing libraries 
import pandas as pd
import numpy as np
 
# importing the dataset
from pipeline_config import cfg
from pathlib import Path

def load_data(interim_data_path : Path):
    df = pd.read_csv(interim_data_path, low_memory = False)

    df.columns = (df.columns
                .str.strip()        # remove leading/trailing whitespace
                .str.rstrip('_'))    # drop any trailing underscores
    return df

# dropping the useless features
def drop_features(df):
    # dropping the features that have high null values, leaking and has low variance
    drop_cols = [
        # Unique identifiers
        'loan_identifier',
        # Near-constant flags (no signal / low variance)
        'modification_flag',
        'servicing_activity_indicator',
        'relocation_mortgage_indicator',
        'high_balance_loan_indicator',
        'borrower_assistance_plan',  # also leaks post-default
        'high_loan_to_value_hltv_refinance_option_indicator',
        'repurchase_make_whole_proceeds_flag',
        'alternative_delinquency_resolution',
        'payment_deferral_modification_event_indicator',
        # Leaky or post-event features
        'loan_payment_history',
        'current_loan_delinquency_status',
        # Nearly-empty (>90% missing)
        'zero_balance_code',
        'upb_at_the_time_of_removal',
    ]

    #dropping the columns from the list 
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    return df 

def convert_to_date_cols(df):

    # converting the date features to YYYYMMDD DATEFORMAT 

    raw_dates = [
        'monthly_reporting_period',
        'origination_date',
        'first_payment_date',
        'maturity_date',
        'zero_balance_effective_date'
    ]
    
    # Zero → NA, then convert to pandas Int64 (so NAs survive)
    df[raw_dates] = df[raw_dates].replace(0, pd.NA).astype('Int64')
    
    date_cols = {
        'monthly_reporting_period': 'report_period_dt',
        'origination_date': 'origination_dt',
        'first_payment_date': 'first_payment_dt',
        'maturity_date': 'maturity_dt',
        'zero_balance_effective_date': 'zero_balance_dt',
    }

    # Parse into datetime, using Int64 to get clean strings
    for raw, dt_col in date_cols.items():
        if raw in df.columns:
            df[dt_col] = (
                df[raw]
                .astype(str)                # "102054" or "<NA>"
                .str.zfill(6)               # "102054"
                .pipe(pd.to_datetime,       # parse month+year
                        format='%m%Y',
                        errors='coerce')
            )
    
    # Derive deltas
    if {'report_period_dt','origination_dt'}.issubset(df.columns):
        df['months_since_origination'] = (
            (df['report_period_dt'].dt.year  - df['origination_dt'].dt.year) * 12 +
            (df['report_period_dt'].dt.month - df['origination_dt'].dt.month)
        )
    if {'first_payment_dt','origination_dt'}.issubset(df.columns):
        df['days_to_first_payment'] = (
            df['first_payment_dt'] - df['origination_dt']
        ).dt.days
    if {'maturity_dt','report_period_dt'}.issubset(df.columns):
        df['days_to_maturity'] = (
            df['maturity_dt'] - df['report_period_dt']
        ).dt.days

    # Drop the raw YYYYMM columns
    df = df.drop(columns=[c for c in raw_dates if c in df.columns])
    return df

def composite_features(df) -> pd.DataFrame:
        # prepare composite features
    # 1. Balance change ratio (captures how much principal has paid down)
    if {'original_upb', 'current_actual_upb'}.issubset(df.columns):
        df['upb_ratio'] = df['current_actual_upb'] / df['original_upb']

    # 2. Min credit score across borrower(s)
    score_cols = [
        'borrower_credit_score_at_origination',
        'co_borrower_credit_score_at_origination'
    ]
    existing_scores = [c for c in score_cols if c in df.columns]
    if existing_scores:
        df['min_credit_score'] = df[existing_scores].min(axis=1)

    # 3. LTV spread (if combined CLTV existed, now dropped)
    if {'original_loan_to_value_ratio_ltv',
        'original_combined_loan_to_value_ratio_cltv'}.issubset(df.columns):
        df['ltv_spread'] = (
            df['original_combined_loan_to_value_ratio_cltv'] -
            df['original_loan_to_value_ratio_ltv']
        )
    # === Impute & Flag Composites ===
    for col in ['upb_ratio', 'min_credit_score', 'ltv_spread']:
        if col in df.columns:
            # 1. Create an “imputed” flag
            df[f"{col}_imputed"] = df[col].isna().astype(int)
            # 2. Fill missing with the column’s median
            median_val = df[col].median()
            df.loc[:, col] = df[col].fillna(median_val)

    return df

def impute_cols(df, drop_thresh: float=0.5, low_thresh: float = 0.05, high_thresh: float = 0.10) -> pd.DataFrame:
    '''
        1. drop_thresh set to 0.5 -> drop any column > than this threshold
        2. For numerical features -> impute with median + flag
        3. for categorical
            - pct_missing <= low_thresh -> impute with mode for this threshold without any flag
            - pct_missing >= high_thresh -> impute with 'missing' + flag
            - otherwise -> impute with mode + flag
    '''
    df = df.copy()

    # dropping the cols with more that drop threshold ( extremely sparse columns)
    min_thresh_na = int((1 - drop_thresh) * len(df))
    df = df.dropna(axis=1, thresh=min_thresh_na)

    # Numeric imputation (median + flag)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            median = df[col].median()
            df[f"{col}_imputed"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(median)

    # Categorical imputation (dynamic)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    miss_pct = df[cat_cols].isna().mean()

    for col in cat_cols:
        pct = miss_pct[col]
        if pct == 0:
            continue
        mode_val = df[col].mode().iloc[0]
        if pct <= low_thresh:
            df.loc[:, col] = df[col].fillna(mode_val)
        elif pct >= high_thresh:
            df.loc[:, f"{col}_imputed"] = df[col].isna().astype(int)
            df.loc[:, col]              = df[col].fillna("Missing")
        else:
            df.loc[:, f"{col}_imputed"] = df[col].isna().astype(int)
            df.loc[:, col]              = df[col].fillna(mode_val)

    return df

def cap_outliers(df, cols: list, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    
    #Clip each column at the given lower and upper quantiles.
    for col in cols:
        lo, hi = df[col].quantile([lower_pct, upper_pct])
        df.loc[:, col] = df[col].clip(lower=lo, upper=hi)
    return df

def log_transform(df, cols: list) -> pd.DataFrame:
    
    #Apply log1p to reduce right skew.
    for col in cols:
        df.loc[:, col] = np.log1p(df[col])
    return df

def standardize(df) -> pd.DataFrame:
    
    # Zero-mean, unit-variance scaling.
    cont_cols = df.select_dtypes(include=['float']).columns.tolist()
    for col in cont_cols:
        mean, std = df[col].mean(), df[col].std()
        df.loc[:, col] = (df[col] - mean) / std

    return df

def prune_col(df) -> pd.DataFrame:
    '''
        1. drop one feature from each highly collinear pair
        2. create composite features that capture joint information 
    '''

    # list of the column to drop based on their high collinear > 0.8
    prune_list = [
        'current_actual_upb',                      # keep original_upb
        'total_principal_current',                 # redundant with original_upb
        'remaining_months_to_maturity',            # redundant with original_loan_term & loan_age
        'remaining_months_to_legal_maturity',      # same story
        'co_borrower_credit_score_at_origination', # collapse into min_score
        'original_combined_loan_to_value_ratio_cltv' # keep original_ltv
    ]

    #dropping them 
    df = df.drop(columns=[col for col in prune_list if col in df.columns])

    return df

def encode_categorical_features(df, cat_thresh = 10) -> pd.DataFrame:
    '''
        -> for each categorical column check for categories 
            - if unique < threshold categories, do one-hot encoding directly
            - if unique > threshold categories, group to top N categories,and remaining to others and the one-hot encode. 
        -> target encode for very high cardinality features.
    '''
    # identify categorical features
    cat_cols = df.select_dtypes(include = ['object', 'category']).columns.tolist()

    for col in cat_cols:

        #check for nuniques values
        n_unique = df[col].nunique(dropna = False)

        if n_unique <= cat_thresh:
            # one-hot encode
            dummies = pd.get_dummies(df[col], prefix=col,dummy_na=False, dtype=int)
            df = pd.concat([df,dummies], axis=1)
            df.drop(columns=[col], inplace=True)

        else:
            # first group to top N and others and then encode
            top_vals = df[col].value_counts().nlargest(cat_thresh).index
            grp_col = f"{col}_grp"
            df[grp_col] = df[col].where(df[col].isin(top_vals), 'Other')

            dummies = pd.get_dummies(df[grp_col], prefix=col, dummy_na=False, dtype=int)
            df = pd.concat([df, dummies], axis=1)

            df.drop(columns=[col, grp_col], inplace=True) 

    if 'property_state' in df.columns:
        # For each row, look up the mean default rate of its state
        df['property_state_target_enc'] = (
            df.groupby('property_state')['default_flag']
            .transform('mean')
        )
        # Drop the original categorical column
        df.drop(columns=['property_state'], inplace=True)


    return df

def save_data(df: pd.DataFrame, processed_data_path: Path) -> None:
    """
    Save the cleaned data to the processed directory.
    """
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_data_path, index=False)

def main():
    # function 

    interim_dir = cfg["data"]["interim_dir"]
    interim_data_path = interim_dir / "credit_risk_data.csv"

    processed_dir = cfg["data"]["processed_dir"]
    processed_data_path = processed_dir / "credit_risk_features.csv"

    print("▶️  Loading the interim data...")
    df = load_data(interim_data_path)    

    print("▶️  Performing clean and drop...")
    # dropping the useless features selected from EDA 
    df = drop_features(df)

    print("▶️  Performing basic transformation...")
    # converting the data columns to DATEFORMAT
    df = convert_to_date_cols(df)

    # lists of columns that are skewed
    skew_cols = [
        'original_upb', 
        'current_actual_upb',
        'debt_to_income_dti', 
        'borrower_credit_score_at_origination',
        'original_interest_rate',
        'current_interest_rate',
        'loan_age'
    ]
    transf_cols = ['original_upb', 'current_actual_upb']

    # Cap outliers at 1st/99th percentile
    df = cap_outliers(df, skew_cols, lower_pct=0.01, upper_pct=0.99)

    # Log-transform heavy skew
    df = log_transform(df, transf_cols)

    print("▶️  Performing Feature construction...")
    # create a composite features combinining features 
    df = composite_features(df)

    print("▶️  Performing imputation for missing values...")
    #imputing the missing columns values 
    df = impute_cols(df, drop_thresh=0.5, low_thresh=0.05, high_thresh=0.10)

    print("▶️  Performing redundancy pruning...")
    #remove one column from high collinear pairs
    df = prune_col(df)

    print("▶️  Performing standarization...")
    #standarize the numerical features 
    df = standardize(df)

    print("▶️  Performing one-hot encoding...")
    # one-hot encode categorical features 
    df = encode_categorical_features(df, 10)

    # saving the processed data
    print("▶️  Saving the processed data...")
    save_data(df, processed_data_path)

    print("✅ Data feature engineering is complete. The processed data is saved to:", processed_data_path)

if __name__ == "__main__":
    main()
    print("Build Features script executed successfully.")     
