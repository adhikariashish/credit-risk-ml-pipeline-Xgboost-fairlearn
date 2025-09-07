import pandas as pd 
import numpy as np
from pipeline_config import cfg
import re
from pathlib import Path
import joblib
import shap
import argparse
from numpy.random import default_rng
import warnings  
# Ignore that specific FutureWarning from NumPy
warnings.filterwarnings(
    "ignore",
    message=".*The NumPy global RNG was seeded.*",
    category=FutureWarning,
)
def load_model(serial: str):
    """
        Load a serialized model bundle by quarter tag(s).
        serial can be:
        - "2024Q4"
        - "2024Q1_2024Q2"
    """
    # Split on underscore to get individual quarter tags
    quarters = serial.split("_")
    
    # Validate each tag
    for q in quarters:
        if not re.fullmatch(r"\d{4}Q[1-4]", q):
            raise ValueError(
                f"Invalid serail format: '{q}'. Expected YYYYQn or YYYYQn_YYYYQn, e.g. 2024Q4 or 2024Q1_2024Q2."
            )

    model_dir = cfg["model"]["output_dir"]
    final_model_path = model_dir/cfg["model"]["output_template"].format(quarter=serial)
    
    bundle = joblib.load(final_model_path)
    preprocessor = bundle.get("preprocessor")
    model = bundle.get("model")
    threshold = bundle.get("threshold")
    feature_names = bundle.get("feature_names")

    return preprocessor, model, threshold, feature_names

def predict(model, X: pd.DataFrame, threshold: float=None):

    '''
        Apply model for the new data and generate probability and binary label for it.
    '''

    # probabilities 
    if hasattr(model, "predict_proba"):
        # calibrated XGBoost 
        probas = model.predict_proba(X)[:, 1]
    else:
        # implement fairlearn ExponentiatedGradient ensemble 
        probas = np.zeros(len(X))
        for m, w in zip(model.predictors_, model.weights_):
            probas += w * m.predict_proba(X)[:, 1]

    # implement threshold 
    cutoff = 0.5 if threshold is None else threshold
    preds = (probas >= cutoff).astype(int)

    return preds, probas

def build_explainer(model, background: pd.DataFrame, rng=None):
    """
        Create a SHAP explainer for the given model using background data.
    """
    try:
        return shap.TreeExplainer(model, data=background, rng=rng)
    except Exception as e:
        # TreeExplainer will raise an ExplainError if the model isn't supported
        # We catch any Exception here to fall back gracefully
        pass

    # If it’s a Fairlearn wrapper, wrap its predict_proba
    if hasattr(model, "predictors_") and hasattr(model, "weights_"):
        def fair_proba(X):
            # Accept either DataFrame or ndarray
            X_arr = X.values if hasattr(X, "values") else X
            p = np.zeros(X_arr.shape[0])
            for m, w in zip(model.predictors_, model.weights_):
                # each m is a base estimator with predict_proba
                p += w * m.predict_proba(X_arr)[:, 1]
            return p
        
        # Build the generic explainer around our fair_proba function
        return shap.Explainer(fair_proba, background, rng=rng)

    # Generic fallback for any model that implements predict_proba
    if hasattr(model, "predict_proba"):
        return shap.Explainer(model.predict_proba, background, rng=rng)

    #If we get here, we can't explain this model
    raise ValueError(f"Cannot build SHAP explainer for model of type {type(model)}")

def explain(explainer, X: pd.DataFrame):
    return explainer(X)

def main():
    
    
    parser = argparse.ArgumentParser(description="Predict and explain with credit-risk model")
    
    parser.add_argument("--serial", required=True, help="Path to serialized model bundle. e.g. 2024Q4")
    
    parser.add_argument("--data", required=True, help="Path to CSV/Parquet of new data")
    
    parser.add_argument("--output-csv", help="Optional path to save predictions")
    
    parser.add_argument("--shap-bg", help="Optional path to CSV of background data for SHAP")
    
    parser.add_argument("--explain", action="store_true", help="Whether to run SHAP explanation")

    parser.add_argument("--shap-frac", type=float, default=0.2, help="Fraction of data to sample for SHAP explanations, e.g. 0.2 for 20%")
    
    args = parser.parse_args()
    
    serial = args.serial
    # Load model and threshold 
    preprocessor, model, threshold, feature_names = load_model(serial)
    print("model has been loaded !!")

    data = args.data
    if not re.fullmatch(r"\d{4}Q[1-4]", data):
        raise ValueError(
            f"\n\n❌ Invalid serial format: '{data}'.\n"
            "✅ Expected format is YYYYQn, where n is 1, 2, 3, or 4, e.g. 2024Q4.\n"
        )
    processed_dir = cfg["data"]["processed_dir"]
    processed_data_path = processed_dir / cfg["templates"]["processed"].format(quarter=data)

    # to have the data for the prediction with the x_features 
    # since we are taking the whole data removing the prediction label and sensitive features 
    # In production when we have to predict we wont have the label column
    print("▶️  Loading the processed data...")
    df = pd.read_csv(processed_data_path, low_memory = False)
    feature_cols = [col for col in df.columns 
                    if col not in ("default_flag", cfg["model"]["fairlearn"]["sensitive_feature"])]
    X_features = df[feature_cols]

    X_preprocessed = preprocessor.transform(X_features)
    X_new = pd.DataFrame(X_preprocessed, columns=feature_names, index=X_features.index)
    
    X_cp = X_new.copy()
    
    preds, probas = predict(model, X_new, threshold)
    X_new["proba"]     = probas
    X_new["pred_label"] = preds

    if args.output_csv:
        out_path = Path(args.output_csv)
    else:
        # default to a predictions folder under project root
        default_dir = cfg["model"]["prediction_dir"]
        default_dir.mkdir(parents=True, exist_ok=True)
        out_path = default_dir / f"{data}_preds.csv"

    X_new.to_csv(out_path, index=False)
    print(f"✅ Saved predictions to {out_path}")
    print("-" * 60)
    print("First few rows of the prediction values !! \n")
    print(X_new[["proba", "pred_label"]].head())

    shap_bg_dir = cfg["data"]["processed_dir"]
    bg_file = args.shap_bg
    if bg_file:
        if not re.fullmatch(r"\d{4}Q[1-4]", bg_file):
            raise ValueError(
                f"\n\n❌ Invalid serial format: '{bg_file}'.\n"
                "✅ Expected format is YYYYQn, where n is 1, 2, 3, or 4, e.g. 2024Q4.\n"
            )
    rng = default_rng(42)
    if args.explain and args.shap_bg:

        # samples for the shap explaination
        frac = args.shap_frac
        total = len(X_cp)
        n_samples = max(1, int(total * frac))  # ensure at least 1
        print(f"🔍 Sampling {n_samples:,} rows ({frac*100:.1f}% of {total:,}) for SHAP")
        shap_x_features = X_cp.sample(n=n_samples, random_state=42)

        bg = pd.read_csv(shap_bg_dir/f"{bg_file}_features_bg.csv")

        bg_pre = preprocessor.transform(bg)
        bg_df = pd.DataFrame(bg_pre, columns=feature_names, index=bg.index)        

        explainer  = build_explainer(model, bg_df, rng=rng)
        shap_vals  = explainer(shap_x_features)
        shap.summary_plot(shap_vals,
            features=shap_x_features,
            feature_names=list(shap_x_features.columns)
    )

if __name__ == "__main__":    
    main()