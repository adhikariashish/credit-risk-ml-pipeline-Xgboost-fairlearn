#!/usr/bin/env bash
set -euo pipefail

# -------- CLI flags --------
QUARTERS=""
CFG_PATH=""
MODEL_DIR=""
PRED_DIR=""
SKIP_INGEST=false
SKIP_FEATURES=false
SKIP_TRAIN=""
TRAIN_QUARTERS=""
SERIAL=""
DATAQ=""
EXPLAIN=0
SHAP_BG=""
SHAP_FRAC=""

usage() {
  cat <<EOF
Usage: $0 --quarters 2024Q4[,2024Q1,...] [--config pipeline_config/config.yaml] [--output-dir models] [--pred-dir predictions] [--skip-train]
       $0 --serial 2024Q4[_2025Q1...] --data 2024Q4 [--explain --shap-bg 2024Q4 --shap-frac 0.10]

Examples:
  $0 --serial 2024Q4 --data 2024Q4
  $0 --serial 2024Q4_2025Q1 --data 2024Q4 --explain --shap-bg 2024Q4 --shap-frac 0.2
EOF
}

# ----- Helper: validate quarter string -----
validate_quarters() {
  local re='^([0-9]{4}Q[1-4])(,([0-9]{4}Q[1-4]))*$'
  [[ $1 =~ $re ]]
}

sanitize_token() {
  local s="$1"
  s="${s//$'\r'/}"                # drop CR
  s="${s//[[:space:]]/}"          # drop all whitespace
  printf '%s' "$s"
}

validate_serial() {
  local s
  s="$(sanitize_token "$1")"
  s="${s//_/,}"         
  validate_quarters "$s"
}


while [[ $# -gt 0 ]]; do
  case "$1" in
    --quarters|--quarter|-q)
      [[ -n "${2:-}" ]] || { echo "❌ --quarters needs a value"; usage; exit 1; }
      QUARTERS="$2"; shift 2 ;;
    --config|-c)
      [[ -n "${2:-}" ]] || { echo "❌ --config needs a path"; usage; exit 1; }
      CFG_PATH="$2"; shift 2 ;;
    --output-dir)
      [[ -n "${2:-}" ]] || { echo "❌ --output-dir needs a path"; usage; exit 1; }
      MODEL_DIR="$2"; shift 2 ;;
    --pred-dir)
      [[ -n "${2:-}" ]] || { echo "❌ --pred-dir needs a path"; usage; exit 1; }
      PRED_DIR="$2"; shift 2 ;;
    --skip-train)
      SKIP_TRAIN=true; shift ;;
    --skip-ingest)
      SKIP_INGEST="true"; shift ;;
    --skip-features)
      SKIP_FEATURES="true"; shift ;;  
    --serial)
      [[ -n "${2:-}" ]] || { echo "❌ --serial needs a value"; usage; exit 1; }
      SERIAL="$(sanitize_token "$2")"; shift 2 ;;
    --data)
      [[ -n "${2:-}" ]] || { echo "❌ --data needs a value (YYYYQn)"; usage; exit 1; }
      DATAQ="$(sanitize_token "$2")"; shift 2 ;;
    --explain)  EXPL=1; shift ;;
    --shap-bg)
      [[ -n "${2:-}" ]] || { echo "❌ --shap-bg needs a value (YYYYQn)"; usage; exit 1; }
      SHAP_BG="$(sanitize_token "$2")"; shift 2 ;;
    --shap-frac)
      [[ -n "${2:-}" ]] || { echo "❌ --shap-frac needs a number in [0,1]"; usage; exit 1; }
      SHAP_FRAC="$(sanitize_token "$2")"; shift 2 ;;   
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "❌ Unknown arg: $1"; usage; exit 1 ;;
  esac
done

echo "====== Credit Risk Pipeline Launcher ======"
echo

# Normalize any CLI-provided value (remove spaces like "2024Q1, 2024Q2")
QUARTERS="${QUARTERS:-}"
if [[ -n "$QUARTERS" ]]; then
  QUARTERS="${QUARTERS//[[:space:]]/}"
fi

# If provided on CLI, validate; if invalid, fall back to interactive
if [[ -n "$QUARTERS" ]]; then
  if validate_quarters "$QUARTERS"; then
    echo "✔️  Using quarters: $QUARTERS"
  else
    echo "⚠️  --quarters '$QUARTERS' is invalid; Try again (e.g. 2024Q4 or 2024Q1,2024Q2)."
    QUARTERS=""
  fi
fi

# If still empty, prompt until valid
if [[ -z "$QUARTERS" ]]; then
  while true; do
    read -p "Enter quarter(s) [e.g. 2024Q4 or 2024Q1,2024Q2]: " QUARTERS
    QUARTERS="${QUARTERS//[[:space:]]/}"
    if validate_quarters "$QUARTERS"; then
      echo "✔️  Using quarters: $QUARTERS"
      break
    else
      echo "❌  Invalid format. Try again (e.g. 2024Q4 or 2024Q1,2024Q2)."
    fi
  done
fi

echo

# Validate CFG_PATH; if missing/invalid, prompt user
CANDIDATE="${CFG_PATH/#\~/$HOME}"
if [[ -f "$CANDIDATE" ]]; then
  CFG_PATH="$CANDIDATE"
  echo "✔️ Using config: $CFG_PATH"
else
  echo "⚠️ Config path not entered !!."
  while true; do
    read -p "Config file [default: pipeline_config/config.yaml]: " INPUT
    INPUT=${INPUT:-pipeline_config/config.yaml}
    INPUT="${INPUT/#\~/$HOME}"
    if [[ -f "$INPUT" ]]; then
      CFG_PATH="$INPUT"
      echo "✔️ Using config: $CFG_PATH"
      break
    else
      echo "❌ Not found: '$INPUT'. Try again."
    fi
  done
fi

echo

# Load config values via Python (so we don’t duplicate paths here)
# We grab processed_dir, model_dir, and predictions_dir from the YAML.
mapfile -t CFG_LINES < <(CFG_PATH="$CFG_PATH" python - <<'PY'
import os, yaml
from pathlib import Path

p = Path(os.environ['CFG_PATH']) # read from env, not shell expansion
with p.open('r', encoding='utf-8') as f:  # force UTF-8
    cfg = yaml.safe_load(f)

print(cfg["data"]["processed_dir"])
print(cfg["model"]["output_dir"])
print(cfg["model"]["prediction_dir"])
print(cfg["model"].get("shap_bg_dir", cfg["data"]["processed_dir"]))
PY
)

# Safety check to avoid unbound variable errors
if ((${#CFG_LINES[@]} != 4)); then
  echo "❌ Failed to read paths from config: $CFG_PATH"
  exit 1
fi

PROCESSED_DIR="${CFG_LINES[0]}"
MODEL_DIR="${CFG_LINES[1]}"
PRED_DIR="${CFG_LINES[2]}"
SHAP_BG_DIR="${CFG_LINES[3]}"

# --- normalize Windows cruft (CR and backslashes) ---
strip_cr() { printf '%s' "$1" | tr -d '\r'; }
to_unix() {
  local p; p="$(strip_cr "$1")"
  printf '%s' "${p//\\//}"   # backslashes -> forward slashes
}

PROCESSED_DIR="$(to_unix "$PROCESSED_DIR")"
MODEL_DIR="$(to_unix "$MODEL_DIR")"
PRED_DIR="$(to_unix "$PRED_DIR")"
SHAP_BG_DIR="$(to_unix "$SHAP_BG_DIR")"

echo "Directories:"
echo "  • Processed data → $PROCESSED_DIR"
echo "  • Model output   → $MODEL_DIR"
echo "  • Predictions    → $PRED_DIR"
echo "  • SHAP background→ $SHAP_BG_DIR"
echo

# Step 1: Data ingestion (raw -> processed)
DEFAULT_CFG="pipeline_config/config.yaml"
echo "▶️  Step 1: Data ingestion"
echo
CFG_ARG=()
if [[ "$CFG_PATH" != "$DEFAULT_CFG" ]]; then
  CFG_ARG=(--config "$CFG_PATH")
fi

# Ingestion
if [[ "$SKIP_INGEST" != "true" ]]; then
    python src/data/make_dataset.py \
        --quarters "$QUARTERS" \
        "${CFG_ARG[@]}"
  echo "✅ Data Ingestion completed !!"
  echo
else
  echo "⏭️  Skipping ingestion — using existing processed files in $PROCESSED_DIR"
  echo
fi

# Feature engineering
# Features
echo "▶️  Step 2: Feature Engineering"
echo
if [[ "$SKIP_FEATURES" != "true" ]]; then
  python src/features/build_features.py \
    --quarters "$QUARTERS" \
    "${CFG_ARG[@]}"
  echo "✅ Feature Engineering completed !!"
  echo
else
  echo "⏭️  Skipping feature engineering — using existing feature files in $PROCESSED_DIR"
  echo
fi

# check if the user wants to retrain the model 
echo "▶️  Step 3: Train & evaluate model"
echo
if [[ -z "$SKIP_TRAIN" ]]; then
    if [[ -t 0 ]]; then  # interactive terminal
        read -p "Re-train model? [y/N]: " RETRAIN_ANS
        RETRAIN_ANS=${RETRAIN_ANS:-N}
        if [[ "$RETRAIN_ANS" =~ ^[Yy]$ ]]; then
            SKIP_TRAIN="false"
            echo " ✅ Let's train some model" 
            echo
        else
            SKIP_TRAIN="true"
            echo " ⏩ Moving forward with existing model."
            echo
        fi
    fi
else
# Non-interactive: default to skip
SKIP_TRAIN="true"
echo "⏩ Moving forward with existing model."
fi

echo

# --- only if we are NOT skipping training, decide training quarters ---
if [[ "$SKIP_TRAIN" != "true" ]]; then
  # If user passed --train-quarters, validate it; otherwise prompt with default = $QUARTERS
    if [[ -n "$TRAIN_QUARTERS" ]]; then
        TRAIN_QUARTERS="${TRAIN_QUARTERS//[[:space:]]/}"  # strip spaces
        if ! validate_quarters "$TRAIN_QUARTERS"; then
            echo "❌ Invalid --train-quarters: '$TRAIN_QUARTERS' (use 2024Q4 or 2024Q1,2024Q2)"; exit 1
        fi
    else
        # Interactive prompt (default to the same set used elsewhere)
        while true; do
            read -p "Which quarter(s) to use for TRAINING? [default: $QUARTERS]: " TRAIN_QUARTERS
            TRAIN_QUARTERS="${TRAIN_QUARTERS//[[:space:]]/}"
            TRAIN_QUARTERS="${TRAIN_QUARTERS:-$QUARTERS}"
            if validate_quarters "$TRAIN_QUARTERS"; then
                echo "✔️  Training on: $TRAIN_QUARTERS"
                break
            else
                echo "❌ Invalid format. Example: 2024Q4 or 2024Q1,2024Q2"
            fi
        done
    fi


    # If you allow skipping features, make sure feature files exist for chosen training quarters
    if [[ "$SKIP_FEATURES" == "true" ]]; then
        IFS=',' read -r -a TQ_ARR <<< "$TRAIN_QUARTERS"
        for Q in "${TQ_ARR[@]}"; do
            Q="${Q//[[:space:]]/}"
            FEAT_FILE="$PROCESSED_DIR/credit_risk_features_${Q}.csv"
            if [[ ! -f "$FEAT_FILE" ]]; then
                echo "❌ Missing features for training: $FEAT_FILE"
                echo " 📁 File format required credit_risk_features_2024Q4.csv . "
                echo "   Remove --skip-features or run build_features.py for $Q first."
                exit 1
            fi
        done
    fi

    # ---- training call uses TRAIN_QUARTERS (not QUARTERS) ----
    python src/models/train_model.py \
        --quarters "$TRAIN_QUARTERS" \
        "${CFG_ARG[@]}" \
        --output-dir "$MODEL_DIR"
else
  echo "⏭️  Skipping training — using existing model(s)"
fi

echo

# ---predict model------
prompt_model_serial() {
echo "📦 Models are stored as:"
  echo "   ${MODEL_DIR}/final_model_<SERIAL>.joblib"
  echo "   e.g., final_model_2024Q4.joblib or final_model_2024Q4_2025Q1.joblib"
  echo

  local RAW REPLY2
  while :; do
    read -rp "Enter model SERIAL (YYYYQn or YYYYQn_YYYYQm...): " RAW
    SERIAL="$(sanitize_token "$RAW")"
    [[ -n "$SERIAL" ]] || { echo "❌ Serial is required."; continue; }

    if ! validate_serial "$SERIAL"; then
      echo "❌ Invalid serial. Use YYYYQn or YYYYQn_YYYYQm..."
      continue
    fi

    MODEL_PATH="${MODEL_DIR}/final_model_${SERIAL}.joblib"
    if [[ -f "$MODEL_PATH" ]]; then
      echo "✅ Found model: $MODEL_PATH"
      export SERIAL MODEL_PATH
      return 0
    fi

    echo "❌ Not found: $MODEL_PATH"
    read -rp "Type '?' to list available models, or enter another serial: " REPLY2
    REPLY2="$(sanitize_token "$REPLY2")"
    if [[ "$REPLY2" == "?" ]]; then
      echo "Available models:"
      ls -1 "${MODEL_DIR}"/final_model_*.joblib 2>/dev/null \
        | sed 's|.*/final_model_||; s|\.joblib$||' || echo "  (none)"
      # loop will re-prompt
    else
      # try the new input immediately on next loop
      RAW="$REPLY2"
      SERIAL="$(sanitize_token "$RAW")"
      # don't duplicate code; let the loop re-validate
    fi
  done
}

SERIAL="$(sanitize_token "${SERIAL:-}")"   # from --serial (may be empty)

if [[ -n "$SERIAL" ]] && validate_serial "$SERIAL"; then
  MODEL_PATH="${MODEL_DIR}/final_model_${SERIAL}.joblib"
  if [[ -f "$MODEL_PATH" ]]; then
    echo "✅ Using model: $MODEL_PATH"
  else
    echo "⚠️ Model not found for --serial '$SERIAL': $MODEL_PATH"
    echo "→ Switching to prompt…"
    prompt_model_serial || exit 1   #  SERIAL + MODEL_PATH
  fi
else
  [[ -n "$SERIAL" ]] && echo "⚠️ Invalid --serial '$SERIAL' (use YYYYQn or YYYYQn_YYYYQm...)"
  prompt_model_serial || exit 1     #  SERIAL + MODEL_PATH
fi

prompt_data_quarter() {
  echo "🗂️ Features files are stored as:"
  echo "   ${PROCESSED_DIR}/credit_risk_features_<QUARTER>.csv"
  echo "   e.g., ${PROCESSED_DIR}/credit_risk_features_2024Q4.csv"
  echo

  local RAW REPLY2
  while :; do
    read -rp "Quarter to score (YYYYQn): " RAW
    DATAQ="$(sanitize_token "$RAW")"                 #  global DATAQ
    [[ -n "$DATAQ" ]] || { echo "❌ Quarter is required."; continue; }

    if ! validate_quarters "$DATAQ"; then
      echo "❌ Invalid quarter. Expected YYYYQn (e.g., 2025Q1)."
      continue
    fi

    DATA_FILE="${PROCESSED_DIR}/credit_risk_features_${DATAQ}.csv"   # global DATA_FILE
    if [[ -f "$DATA_FILE" ]]; then
      echo "✅ Found features: $DATA_FILE"
      return 0
    fi

    echo "❌ Not found: $DATA_FILE"
    read -rp "Type '?' to list available quarters, or enter another quarter: " REPLY2
    REPLY2="$(sanitize_token "$REPLY2")"
    if [[ "$REPLY2" == "?" ]]; then
      echo "Available quarters:"
      ls -1 "${PROCESSED_DIR}"/credit_risk_features_????Q?.csv 2>/dev/null \
        | sed -E 's#.*/([0-9]{4}Q[1-4])_features\.csv#\1#' \
        | sort -u || echo "  (none)"
      # loop will re-prompt
    else
      RAW="$REPLY2"    # try the new input on next loop
    fi
  done
}

DATAQ="$(sanitize_token "${DATAQ:-}")"

if [[ -n "$DATAQ" ]] && validate_quarters "$DATAQ"; then
  DATA_FILE="${PROCESSED_DIR}/credit_risk_features_${DATAQ}.csv"
  if [[ -f "$DATA_FILE" ]]; then
    echo "✅ Using features: $DATA_FILE"
  else
    echo "⚠️ Features not found for --data '$DATAQ': $DATA_FILE"
    echo "→ Switching to prompt…"
    prompt_data_quarter || exit 1
  fi
else
  [[ -n "$DATAQ" ]] && echo "⚠️ Invalid --data '$DATAQ' (expected YYYYQn)"
  prompt_data_quarter || exit 1
fi

echo "✅ Predicting flag for : $DATA_FILE"

predict_loan(){
  local PY="${PYTHON:-python}"
  local SCRIPT="${PREDICT_SCRIPT:-src/models/predict_model.py}"

  # ensure output dir exists
  mkdir -p "$PRED_DIR"

  # --- model serial ---
  prompt_model_serial || return 1   # sets $SERIAL and $MODEL_PATH    


}
# ---------- SHAP options + run predict ----------
EXPL_ARGS=()

if [[ "${EXPL:-0}" -eq 1 ]]; then
  # --- SHAP BG quarter ---
  if [[ -n "${SHAP_BG:-}" ]]; then
    SHAP_BG="$(sanitize_token "$SHAP_BG")"
    validate_quarters "$SHAP_BG" || { echo "❌ --shap-bg must be YYYYQn"; exit 1; }
  else
    while :; do
      read -rp "Background quarter for SHAP (YYYYQn): " SHAP_BG
      SHAP_BG="$(sanitize_token "$SHAP_BG")"
      validate_quarters "$SHAP_BG" && break || echo "❌ Invalid quarter."
    done
  fi
  BG_FILE="${SHAP_BG_DIR}/${SHAP_BG}_features_bg.csv"
  [[ -f "$BG_FILE" ]] || { echo "❌ Missing SHAP BG file: $BG_FILE"; exit 1; }

  # --- SHAP fraction ---
  if [[ -n "${SHAP_FRAC:-}" ]]; then
    SHAP_FRAC="$(sanitize_token "$SHAP_FRAC")"
  else
    read -rp "Fraction of rows for SHAP (0–1, default 0.10): " SHAP_FRAC
    SHAP_FRAC="${SHAP_FRAC:-0.10}"
    SHAP_FRAC="$(sanitize_token "$SHAP_FRAC")"
  fi
  [[ "$SHAP_FRAC" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]] || { echo "❌ Invalid fraction: $SHAP_FRAC"; exit 1; }

  EXPL_ARGS=( --explain --shap-bg "$SHAP_BG" --shap-frac "$SHAP_FRAC" )
fi

# ---------- run predict_model.py ----------
PY="${PYTHON:-python}"
SCRIPT="${PREDICT_SCRIPT:-src/models/predict_model.py}"
OUT_PATH="${PRED_DIR}/${DATAQ}_preds.csv"

"$PY" "$SCRIPT" \
  --serial "$SERIAL" \
  --data "$DATAQ" \
  --output-csv "$OUT_PATH" \
  ${EXPL_ARGS[@]+"${EXPL_ARGS[@]}"} || exit $?

echo "🎉 All done! Models in '$MODEL_DIR', predictions in '$PRED_DIR'."