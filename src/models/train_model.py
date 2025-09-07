
# importing libraries 
import pandas as pd
import numpy as np
 
# importing the dataset
from pathlib import Path
import argparse
import re
import warnings
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, make_scorer, recall_score, precision_score, accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, ParameterGrid
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

warnings.filterwarnings(
    "ignore",
    message=".*FrozenEstimator.*sample_weight.*",
    category=UserWarning,
    module="sklearn.calibration"
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from imblearn.over_sampling import SMOTE
import joblib
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

_boot = argparse.ArgumentParser(add_help=False)
_boot.add_argument("-c", "--config")
cfg_args, _ = _boot.parse_known_args()   # ignore all other args for now

if cfg_args.config:
    os.environ["CR_CONFIG_PATH"] = cfg_args.config

from pipeline_config import cfg

def load_data(quarter: str):

    # Validate the quarter format: four digits + ‘Q’ + 1–4
    if not re.fullmatch(r"\d{4}Q[1-4]", quarter):
        raise ValueError(
            f"\n\n❌ Invalid quarter format: '{quarter}'.\n"
            "✅ Expected format is YYYYQn, where n is 1, 2, 3, or 4, e.g. 2024Q4.\n"
        )

    processed_dir = cfg["data"]["processed_dir"]
    processed_data_path = processed_dir / cfg["templates"]["processed"].format(quarter=quarter)

    print("▶️  Loading the processed data...")
    df = pd.read_csv(processed_data_path, low_memory = False)

    # sensitive features for fairlearn
    fl_dir = cfg["model"]["fairlearn"]
    fl_cols = fl_dir["sensitive_feature"]

    # get a list of fairlearn columns 
    if isinstance(fl_cols, (list, tuple)):
        sf_cols = list(fl_cols)
    else:
        sf_cols = [fl_cols]

    # Extract and copy
    if len(sf_cols) == 1:
        # single column → Series
        sf = df[sf_cols[0]].copy()
    else:
        # multiple columns → DataFrame
        sf = df[sf_cols].copy()

    # seperating the target column
    y = df["default_flag"]
    X = df.drop(columns=sf_cols + ["default_flag"])

    # saving random sample for SHAP background set to reflect same feature space 
    bg = X.sample(n=500, random_state=42)
    bg.to_csv(processed_dir/f"{quarter}_features_bg.csv", index=False)

    
    return X, y, sf

def load_multiple_quarters(quarter_str: str):
    """
    Accepts either:
      - "2024Q4"
      - "2024Q1,2024Q2,2024Q4"
    Splits on commas, loads each quarter, and concatenates.
    Returns concatenated (X_all, y_all, sf_all).
    """
    # Split and strip tags
    quarters = [q.strip() for q in quarter_str.split(",") if q.strip()]

    X_list, y_list, sf_list = [], [], []
    for q in quarters:
        X, y, sf = load_data(q)
        X_list.append(X)
        y_list.append(y)
        sf_list.append(sf)

    # Concatenate along the row axis
    X_all  = pd.concat(X_list, axis=0, ignore_index=True)
    y_all  = pd.concat(y_list, axis=0, ignore_index=True)
    sf_all = pd.concat(sf_list, axis=0, ignore_index=True)

    print(f"✅ Combined data from quarters: {', '.join(quarters)}")
    print(f"   → Total samples: {X_all.shape[0]}")
    return X_all, y_all, sf_all    

def split_data (X, y, sf):

    seed = cfg["model"]["random_seed"]

    # Splitting the data into train and test 
    # 60/20/20 stratified
    X_tv, X_test, y_tv, y_test, sf_tv, sf_test = train_test_split(
        X, y, sf, test_size=0.2, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val, sf_train, sf_val = train_test_split(
        X_tv, y_tv, sf_tv, test_size=0.25, stratify=y_tv, random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, sf_train, sf_val, sf_test

# preprocessor for any missing values after concating different values or change in values 
def build_preprocessor(X):
    num_cols = X.select_dtypes(exclude=["object", "category", "bool"]).columns
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, list(num_cols)),
        ("cat", cat_pipe, list(cat_cols)),
    ], remainder="drop")


def resample_train_data(X_train, y_train, sf_train):

    dfs = []
    for group in np.unique(sf_train):
        # select subgroup
        mask = (sf_train == group)
        X_g = X_train[mask]
        y_g = y_train[mask]
        
        # only oversample if positives exist
        if y_g.sum() > 0 and y_g.sum() < len(y_g):
            sm = SMOTE(random_state=42)
            X_res_g, y_res_g = sm.fit_resample(X_g, y_g)
        else:
            # no SMOTE needed if group has only one class
            X_res_g, y_res_g = X_g, y_g
        
        # build a DataFrame to carry group labels
        df_g = pd.DataFrame(X_res_g)
        df_g["_target"] = y_res_g
        df_g["_sf"] = group
        dfs.append(df_g)

    # combine all groups
    df_res = pd.concat(dfs, ignore_index=True)

    # split back out
    X_res = df_res.drop(columns=["_target", "_sf"]).values
    y_res = df_res["_target"].values
    sf_res = df_res["_sf"].values

    return X_res, y_res, sf_res

def train_baseline (X_train, y_train):

    xgb_cfg = cfg["model"]["xgboost"]

    xgb_model = XGBClassifier(
        eval_metric="logloss",
        random_state=cfg["model"]["random_seed"],
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
    ) 
    
    xgb_model.fit(X_train, y_train)

    return xgb_model

def evaluate(name, model, X, y, threshold: float=None):
    """
    Evaluates either a standard classifier (with predict_proba)
    or a Fairlearn ExponentiatedGradient wrapper (without).
    """
    # Get probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        # assume ExponentiatedGradient with model.predictors_ & .weights_
        proba = np.zeros(len(X))
        for mdl, w in zip(model.predictors_, model.weights_):
            proba += w * mdl.predict_proba(X)[:, 1]

    if threshold is None:
        preds = model.predict(X)
    else:
        preds = (proba >= threshold).astype(int)

    print(f"\n--- {name} ---")
    print(f"ROC AUC:           {roc_auc_score(y, proba):.4f}")
    print(f"Avg Precision:    {average_precision_score(y, proba):.4f}")
    print(classification_report(y, preds, digits=4))


def tune_hyperparams(X_train, y_train):

    seed = cfg["model"]["random_seed"]
    tune_cfg = cfg["model"]["tuning"]
    cv_cfg = cfg["model"]["tuning"]["cv_folds"]
    if cfg["model"]["optimize_for"] == "recall":
        scoring = make_scorer(recall_score)
    else:
        scoring = "average_precision"

    base_model = XGBClassifier(
        eval_metric = "logloss",
        random_state = seed,
        n_jobs=1,
        tree_method="hist"
    )

    # building grid around the config values for tunning
    param_grid = tune_cfg["grid"]

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv = cv_cfg,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )

    n_candidates = len(list(ParameterGrid(param_grid)))
    total_fits = n_candidates * cv_cfg

    print(f"▶️  Running tuned model… ({cv_cfg} folds × {n_candidates} candidates = {total_fits} fits)")
    with tqdm_joblib(tqdm(total=total_fits, desc="GridSearch", unit="fit")):
        search.fit(X_train, y_train)

    print("▶▶  Best parameters : " , search.best_params_)

    return search.best_estimator_

def calibrate(model, X_val, y_val):
    method = cfg["model"].get("calibration_method", "isotonic")
    frozen = FrozenEstimator(model)
    calib = CalibratedClassifierCV(
        estimator=frozen,
        method=method,
        cv=3
    )
    calib.fit(X_val, y_val)
    return calib

def train_fairlearn_model(cal_model, X_train, y_train, sf_train, X_val, y_val, sf_val, sel_threshold, fl_dir):
    """
        Here we will wrap `cal_model` in a Fairlearn ExponentiatedGradient reduction,
        fit it on (X_train, y_train) under the specified constraint,
        and prints selection‐rate gaps on the validation set.
        
        Returns:
            fair_clf: the fitted Fairlearn-wrapped classifier

        we picked property_state as sensitive feature since there was no gender or race in the dataset.
    """
    eps = fl_dir["epsilon"]
    constr = fl_dir["constraint"]

    # choose constraint
    if constr == "demographic_parity":
        constraint = DemographicParity(difference_bound=eps)
    else:
        constraint = EqualizedOdds(difference_bound=eps)

    # Build the reduction wrapper
    fair_clf = ExponentiatedGradient(
        estimator=cal_model,
        constraints=constraint,
        max_iter=50,
        eps=eps
    )

    print(f"▶️  Fitting Fairlearn reduction (ε={eps}, constraint={constr})…")
    fair_clf.fit(
        X_train, y_train,
        sensitive_features=sf_train
    )

    # Validate fairness on val set
    print("▶️  Validation performance & fairness:")
    # calibrated model preds
    # cal_preds = (cal_model.predict_proba(X_val)[:,1] >= sel_threshold).astype(int)
    proba_cal = cal_model.predict_proba(X_val)[:, 1]
    cal_preds = (proba_cal >= sel_threshold).astype(int)

    # Fair model preds
    # 1) Initialize an array of zeros
    proba_fair = np.zeros(len(X_val))

    for mdl, w in zip(fair_clf.predictors_, fair_clf.weights_):
        proba_fair += w * mdl.predict_proba(X_val)[:, 1]
    fair_preds = (proba_fair >= sel_threshold).astype(int)

    metrics = ["Accuracy", "ROC-AUC", "Log-loss"]
    base_vals = [
        accuracy_score(y_val, cal_preds),
        roc_auc_score(y_val, proba_cal),
        log_loss(y_val, proba_cal)
    ]
    fair_vals = [
        accuracy_score(y_val, fair_preds),
        roc_auc_score(y_val, proba_fair),
        log_loss(y_val, proba_fair)
    ]

    df_compare = pd.DataFrame({
        "Metric": metrics,
        "Base Model": base_vals,
        "Fairlearn Model": fair_vals
    })

    print("Validation Performance Comparison \n", df_compare)

    # 3) Compute selection‐rate gaps
    from fairlearn.metrics import MetricFrame, selection_rate
    mf_base = MetricFrame(
        metrics=selection_rate,
        y_true=y_val, y_pred=cal_preds,
        sensitive_features=sf_val
    )
    mf_fair = MetricFrame(
        metrics=selection_rate,
        y_true=y_val, y_pred=fair_preds,
        sensitive_features=sf_val
    )


    print(f"Base gap: {mf_base.difference():.4f}")
    print(f"Fair gap: {mf_fair.difference():.4f}")
    print("-" * 60)

    base_acc = base_vals[0]
    fair_acc = fair_vals[0]
    base_gap = mf_base.difference()
    fair_gap = mf_fair.difference()

    metrics = {
        "base_acc":  base_acc,
        "fair_acc":  fair_acc,
        "base_gap":  base_gap,
        "fair_gap":  fair_gap
    }

    return fair_clf, metrics

def choose_model(base_acc, base_gap, fair_acc, fair_gap, sel_cfg):
    if sel_cfg["rule"] == "constraint":
        return "fair" if (fair_acc >= sel_cfg["min_accuracy"]
                         and fair_gap <= sel_cfg["max_gap"]) else "base"
    else:
        score_base = base_acc - sel_cfg["lambda"]*base_gap
        score_fair = fair_acc - sel_cfg["lambda"]*fair_gap
        return "fair" if score_fair > score_base else "base"

def parse_args():
    p = argparse.ArgumentParser(description="Processed loan data by quarter")
    p.add_argument(
        "--quarters", "--quarter" "-q",
        required=True,
        help="Quarter tag to process, e.g. 2024Q4"
    )
    p.add_argument("--output-dir", "-o", help="Path to Output folder for trained model (optional)")
    return p.parse_args()

def main():

    args = parse_args()

    quarter = args.quarters    

    X, y, sf = load_multiple_quarters(quarter)

    print("▶️  Splitting the data for training model...")
    X_train, X_val, X_test, y_train, y_val, y_test, sf_train, sf_val, sf_test = split_data(X, y, sf)

    # copy for reference 
    X_train_raw, X_val_raw, X_test_raw = X_train.copy(), X_val.copy(), X_test.copy()

    #preprocess before performing smote, flaging the nan values which throws error for smote function
    # performing pre-process for X values 
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)
    X_train_pre = preprocessor.transform(X_train_raw)
    X_val_pre = preprocessor.transform(X_val_raw)
    X_test_pre = preprocessor.transform(X_test_raw)

    # Feature names for SHAP/debug
    try:
        feat_names = preprocessor.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_train_pre.shape[1])]


    # dataframes for the smote function 
    X_train = pd.DataFrame(X_train_pre, index=X_train.index, columns=feat_names)
    X_val = pd.DataFrame(X_val_pre, index=X_val.index, columns=feat_names)
    X_test = pd.DataFrame(X_test_pre, index=X_test.index, columns=feat_names)

    if cfg["model"].get("resample", None) == "smote":
            seed = cfg["model"]["random_seed"]
            X_res, y_res, sf_res = resample_train_data(X_train, y_train, sf_train)
    else:
        X_res, y_res, sf_res = X_train, y_train, sf_train

    # --------------------------Training baseline model--------------------------------------------
    print("▶️  Running baseline model...")
    print("="*60)
    base_model = train_baseline(X_res, y_res)
    print("▶️  Getting validation score for baseline model...")
    evaluate("Validation score for baseliine model : ", base_model, X_val, y_val)
    print("="*60)

    # --------------------------Hyperparameter tunning --------------------------------------------
    # tunning the model 
    print("▶️  Running tuned model...")
    print("="*60)
    tuned_model = tune_hyperparams(X_res, y_res)
    print("▶️  Getting validation score for tuned model...")
    evaluate("Validation score for tuned model : ", tuned_model, X_val, y_val)
    print("="*60)

    # --------------------------Calibrating trained model ------------------------------------------
    # calibrating the tuned model
    print("▶️  Calibrating the tuned model...")
    print("="*60)
    if cfg["model"].get("do_calibration", False):
        cal_model = calibrate(tuned_model, X_val, y_val)
        evaluate(" Validation score for calibrated model : ", cal_model, X_val, y_val)

    # --------------------------Threshold selection for model --------------------------------------

    # computing full precision-recall tradeoff on the validation set using calibrated model
    proba_val = cal_model.predict_proba(X_val)[:, 1]

    # implementing quantile based cutoff
    # computing cutoff for top k% as defined in config
    pct_grid      = cfg["model"]["threshold"]["pct_grid"]
    precision_min = cfg["model"]["threshold"].get("precision_min", 0.0)
    pct_max       = cfg["model"]["threshold"].get("pct_max", 1.0)

    results = []
    for pct in pct_grid:
        # skip any pct beyond the max investigation budget
        if pct > pct_max:
            continue

        # compute the cutoff for this top‐pct
        thr   = np.percentile(proba_val, 100 * (1 - pct))
        preds = (proba_val >= thr).astype(int)
        prec  = precision_score(y_val, preds, zero_division=0)
        rec   = recall_score(y_val, preds)
        results.append({
            "top_pct":  pct,
            "threshold": thr,
            "precision": prec,
            "recall":    rec
        })

    df_grid = pd.DataFrame(results)

    # filter by pct_max only
    filtered = df_grid[df_grid.top_pct <= pct_max]
    print(f"\nCandidates with top_pct ≤ {pct_max}:")
    print(filtered.to_string(index=False))

    candidates = filtered[filtered.precision >= precision_min]
    if candidates.empty:
        print(f"\n⚠️  No slice meets precision ≥ {precision_min}; falling back to pct_max only")
        best = filtered.loc[filtered.recall.idxmax()]
    else:
        best = candidates.loc[candidates.recall.idxmax()]

    sel_pct = best.top_pct
    sel_threshold = best.threshold

    print(f"\n ▶▶▶ Chosen top_pct: {sel_pct:.3f} (≤ {pct_max}), threshold: {sel_threshold:.6f}")
    print(f" ▶▶▶ Precision: {best.precision:.4f} (≥ {precision_min}), Recall: {best.recall:.4f}")
    print("="*60)

    # --------------------------fairlearn wrapping -------------------------------------------------
    fl_dir = cfg["model"]["fairlearn"]
    fair_clf, fair_metrics = train_fairlearn_model(
        cal_model,
        X_train, y_train, sf_train,
        X_val, y_val, sf_val,
        sel_threshold,
        fl_dir
    )

    sel_cfg = cfg["model"]["selection"]

    base_acc = fair_metrics["base_acc"]
    fair_acc = fair_metrics["fair_acc"]
    base_gap = fair_metrics["base_gap"]
    fair_gap = fair_metrics["fair_gap"]

    chosen = choose_model(base_acc, base_gap, fair_acc, fair_gap, sel_cfg)

    print("→ Selected model:", chosen)

    # --------------------------Final retrain & testing --------------------------------------------


    print("▶️  Retraining the Chosen model on train+val...")
    print("="*60)
    X_tv = pd.concat([X_train, X_val], axis=0)
    y_tv = pd.concat([y_train, y_val], axis=0)
    sf_tv = pd.concat([sf_train, sf_val], axis=0)

    if chosen == "fair":
        # build constraint object
        fl = cfg["model"]["fairlearn"]
        if fl["constraint"] == "demographic_parity":
            constraint = DemographicParity(difference_bound=fl["epsilon"])
        else:
            constraint = EqualizedOdds(difference_bound=fl["epsilon"])
        # retrain Fairlearn on full train+val
        final_model = ExponentiatedGradient(
            estimator=cal_model,           # your calibrated base
            constraints=constraint,
            max_iter=50,
            eps=fl["epsilon"]
        )
        final_model.fit(
            X_tv, y_tv,
            sensitive_features=sf_tv
        )
    else:
        # simply refit your calibrated base on train+val
        final_model = cal_model
        final_model.fit(X_tv, y_tv)

    # 5) Evaluate on test set
    print("\n▶️  Final Test Evaluation —")


    if chosen.lower() == "fair":
        # reconstruct ensemble probs
        proba_test = np.zeros(len(X_test))
        for mdl, w in zip(final_model.predictors_, final_model.weights_):
            proba_test += w * mdl.predict_proba(X_test)[:, 1]
    else:
        proba_test = final_model.predict_proba(X_test)[:, 1]

    preds_test = (proba_test >= sel_threshold).astype(int)

    print("\n--- Test set results @ threshold = %.2f ---" % sel_threshold)
    print(f"Precision : {precision_score(y_test, preds_test):.4f}")
    print(f"Recall    : {recall_score(y_test, preds_test):.4f}")
    evaluate(f"Score for {chosen} model:", final_model, X_test, y_test, threshold=sel_threshold)
    print("=" * 60)

    # serializing the model 
    quarters = [q.strip() for q in args.quarters.split(",") if q.strip()]
    qt = "_".join(quarters)

    # if path given save in the given path else in the default 
    out_dir = Path(args.output_dir) if hasattr(args, "output_dir") and args.output_dir else Path(cfg["model"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg["model"]["output_template"].format(quarter=qt)

    # Bundle both into one file
    joblib.dump(
        {
        "preprocessor": preprocessor,    
        "model":     final_model,
        "threshold": sel_threshold,
        "feature_names": feat_names
        },
        out_path
    )
    print(f"▶️  Serialized model + threshold to : models/final_model_{qt}")

if __name__ == "__main__":
    main()
    print("🍾 Train model script executed successfully.")     
