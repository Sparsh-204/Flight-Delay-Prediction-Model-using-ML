import argparse
import importlib
import os
import sys
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from weather_features import enrich_weather_features

if __name__ == "__main__":
    sys.modules.setdefault("flight_delay_model", sys.modules[__name__])

warnings.filterwarnings("ignore")

try:
    XGBClassifier: Any = importlib.import_module("xgboost").XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGB_AVAILABLE = False


RANDOM_STATE = 42
MODEL_OUTPUT_PATH = "flight_delay_model.joblib"
DEFAULT_MAX_ROWS = 250000
DELAY_THRESHOLD_MINUTES = 15
WEATHER_FEATURE_COLUMNS = [
    "weather_wind",
    "weather_visibility",
    "weather_rain",
    "weather_storm",
]
BASE_INPUT_COLUMNS = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "DEPARTURE_TIME",
    "DISTANCE",
]


def load_data(csv_path: str, max_rows: int | None = DEFAULT_MAX_ROWS) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    use_columns = BASE_INPUT_COLUMNS + ["DEPARTURE_DELAY"]
    read_kwargs = {"usecols": use_columns, "low_memory": False}
    if max_rows:
        read_kwargs["nrows"] = max_rows

    return pd.read_csv(csv_path, **read_kwargs)


def parse_time_value(value) -> tuple[float, float]:
    if pd.isna(value):
        return np.nan, np.nan

    try:
        raw_value = int(float(value))
        normalized = f"{raw_value:04d}"[-4:]
        hour = min(int(normalized[:2]), 23)
        minute = min(int(normalized[2:]), 59)
        total_minutes = hour * 60 + minute
        return total_minutes, hour
    except Exception:
        return np.nan, np.nan


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    parsed_times = enriched["DEPARTURE_TIME"].apply(parse_time_value)

    enriched["DEPARTURE_MINS"] = parsed_times.apply(lambda item: item[0])
    enriched["DEPARTURE_HOUR"] = parsed_times.apply(lambda item: item[1])
    enriched["DEPARTURE_HOUR"] = enriched["DEPARTURE_HOUR"].fillna(
        pd.to_numeric(enriched["DEPARTURE_TIME"], errors="coerce").floordiv(100)
    )

    enriched["hour_sin"] = np.sin(2 * np.pi * enriched["DEPARTURE_HOUR"] / 24.0)
    enriched["hour_cos"] = np.cos(2 * np.pi * enriched["DEPARTURE_HOUR"] / 24.0)
    enriched["is_rush"] = enriched["DEPARTURE_HOUR"].between(16, 19, inclusive="both").astype(int)
    enriched["is_weekend"] = (enriched["DAY_OF_WEEK"] >= 6).astype(int)
    enriched["TIME_BUCKET"] = pd.cut(
        enriched["DEPARTURE_HOUR"],
        bins=[-1, 5, 11, 17, 23],
        labels=["overnight", "morning", "afternoon", "evening"],
    ).astype(str)
    enriched["ROUTE"] = (
        enriched["ORIGIN_AIRPORT"].astype(str) + "_" + enriched["DESTINATION_AIRPORT"].astype(str)
    )

    return enriched


def ensure_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    defaults = {
        "weather_wind": np.nan,
        "weather_visibility": np.nan,
        "weather_rain": 0.0,
        "weather_storm": 0.0,
    }
    for column, default_value in defaults.items():
        if column not in enriched.columns:
            enriched[column] = default_value
    return enriched


def build_feature_store(train_df: pd.DataFrame) -> dict:
    global_delay_mean = float(train_df["DEPARTURE_DELAY"].mean())

    return {
        "global_delay_mean": global_delay_mean,
        "origin_traffic": train_df["ORIGIN_AIRPORT"].value_counts().to_dict(),
        "dest_traffic": train_df["DESTINATION_AIRPORT"].value_counts().to_dict(),
        "route_traffic": train_df["ROUTE"].value_counts().to_dict(),
        "airline_delay_rate": train_df.groupby("AIRLINE")["DEPARTURE_DELAY"].mean().to_dict(),
        "origin_avg_delay": train_df.groupby("ORIGIN_AIRPORT")["DEPARTURE_DELAY"].mean().to_dict(),
        "dest_avg_delay": train_df.groupby("DESTINATION_AIRPORT")["DEPARTURE_DELAY"].mean().to_dict(),
        "route_avg_delay": train_df.groupby("ROUTE")["DEPARTURE_DELAY"].mean().to_dict(),
        "origin_freq": train_df["ORIGIN_AIRPORT"].value_counts(normalize=True).to_dict(),
        "dest_freq": train_df["DESTINATION_AIRPORT"].value_counts(normalize=True).to_dict(),
    }


def apply_feature_store(df: pd.DataFrame, feature_store: dict) -> pd.DataFrame:
    enriched = add_time_features(df)
    enriched = ensure_weather_columns(enriched)
    global_delay_mean = feature_store["global_delay_mean"]

    enriched["ORIGIN_TRAFFIC"] = enriched["ORIGIN_AIRPORT"].map(feature_store["origin_traffic"]).fillna(0)
    enriched["DEST_TRAFFIC"] = enriched["DESTINATION_AIRPORT"].map(feature_store["dest_traffic"]).fillna(0)
    enriched["ROUTE_TRAFFIC"] = enriched["ROUTE"].map(feature_store["route_traffic"]).fillna(0)
    enriched["AIRLINE_DELAY_RATE"] = (
        enriched["AIRLINE"].map(feature_store["airline_delay_rate"]).fillna(global_delay_mean)
    )
    enriched["ORIGIN_AVG_DELAY"] = (
        enriched["ORIGIN_AIRPORT"].map(feature_store["origin_avg_delay"]).fillna(global_delay_mean)
    )
    enriched["DEST_AVG_DELAY"] = (
        enriched["DESTINATION_AIRPORT"].map(feature_store["dest_avg_delay"]).fillna(global_delay_mean)
    )
    enriched["ROUTE_AVG_DELAY"] = enriched["ROUTE"].map(feature_store["route_avg_delay"]).fillna(global_delay_mean)
    enriched["ORIGIN_AIRPORT_FREQ"] = enriched["ORIGIN_AIRPORT"].map(feature_store["origin_freq"]).fillna(0)
    enriched["DESTINATION_AIRPORT_FREQ"] = enriched["DESTINATION_AIRPORT"].map(feature_store["dest_freq"]).fillna(0)

    return enriched


def build_feature_lists(include_weather: bool) -> tuple[list[str], list[str], list[str]]:
    numeric_columns = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "DEPARTURE_TIME",
        "DEPARTURE_MINS",
        "DEPARTURE_HOUR",
        "hour_sin",
        "hour_cos",
        "DISTANCE",
        "is_rush",
        "is_weekend",
        "ORIGIN_TRAFFIC",
        "DEST_TRAFFIC",
        "ROUTE_TRAFFIC",
        "AIRLINE_DELAY_RATE",
        "ORIGIN_AVG_DELAY",
        "DEST_AVG_DELAY",
        "ROUTE_AVG_DELAY",
        "ORIGIN_AIRPORT_FREQ",
        "DESTINATION_AIRPORT_FREQ",
    ]
    if include_weather:
        numeric_columns.extend(WEATHER_FEATURE_COLUMNS)

    categorical_columns = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "TIME_BUCKET"]
    feature_columns = numeric_columns + categorical_columns
    return feature_columns, numeric_columns, categorical_columns


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_store: dict,
    include_weather: bool,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    transformed = apply_feature_store(df, feature_store)
    feature_columns, numeric_columns, categorical_columns = build_feature_lists(include_weather)

    for column in feature_columns:
        if column not in transformed.columns:
            transformed[column] = np.nan if column in numeric_columns else "missing"

    return transformed[feature_columns], feature_columns, numeric_columns, categorical_columns


def build_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    use_smote: bool,
    model_type: str,
):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    if model_type == "xgboost":
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not installed. Run `pip install xgboost` or use --model-type randomforest.")
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        parameter_grid = {
            "clf__n_estimators": [200, 300],
            "clf__max_depth": [4, 6, 8],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        }
    else:
        classifier = RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        parameter_grid = {
            "clf__n_estimators": [200, 300],
            "clf__max_depth": [10, 14, 18],
            "clf__min_samples_leaf": [1, 2, 4],
        }

    steps = [("preproc", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("clf", classifier))

    return ImbPipeline(steps=steps), parameter_grid


def choose_best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.30, 0.71, 0.05):
        predictions = (y_prob >= threshold).astype(int)
        current_score = f1_score(y_true, predictions)
        if current_score > best_score:
            best_threshold = float(threshold)
            best_score = float(current_score)

    return best_threshold, best_score


class FlightDelayPredictor:
    def __init__(
        self,
        model,
        feature_store: dict,
        feature_columns: list[str],
        numeric_columns: list[str],
        categorical_columns: list[str],
        threshold: float,
        use_weather_features: bool,
    ):
        self.model = model
        self.feature_store = feature_store
        self.feature_columns = feature_columns
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.threshold = threshold
        self.use_weather_features = use_weather_features

    def prepare_input(self, df: pd.DataFrame, include_weather: bool = False) -> pd.DataFrame:
        missing_columns = [column for column in BASE_INPUT_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        prepared = df.copy()
        if self.use_weather_features and include_weather:
            prepared = enrich_weather_features(prepared, airport_column="ORIGIN_AIRPORT")
        else:
            prepared = ensure_weather_columns(prepared)

        feature_matrix, _, _, _ = prepare_feature_matrix(
            prepared,
            feature_store=self.feature_store,
            include_weather=self.use_weather_features,
        )
        return feature_matrix

    def predict_proba(self, df: pd.DataFrame, include_weather: bool = False) -> np.ndarray:
        features = self.prepare_input(df, include_weather=include_weather)
        return self.model.predict_proba(features)[:, 1]

    def predict(self, df: pd.DataFrame, include_weather: bool = False) -> np.ndarray:
        probabilities = self.predict_proba(df, include_weather=include_weather)
        return (probabilities >= self.threshold).astype(int)


FlightDelayPredictor.__module__ = "flight_delay_model"


def train_model(
    csv_path: str = "flights.csv",
    model_output_path: str = MODEL_OUTPUT_PATH,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    enable_weather: bool = False,
    model_type: str = "randomforest",
):
    print("Loading data...")
    df = load_data(csv_path=csv_path, max_rows=max_rows)
    df = df.dropna(subset=["DEPARTURE_DELAY"]).copy()
    df["IS_DELAYED"] = (df["DEPARTURE_DELAY"] > DELAY_THRESHOLD_MINUTES).astype(int)

    print(f"Rows loaded: {len(df):,}")
    print("Delay distribution:")
    print(df["IS_DELAYED"].value_counts(normalize=True).rename("share"))

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["IS_DELAYED"],
        random_state=RANDOM_STATE,
    )

    if enable_weather:
        print("Adding weather features to train/test data...")
        train_df = enrich_weather_features(train_df, airport_column="ORIGIN_AIRPORT")
        test_df = enrich_weather_features(test_df, airport_column="ORIGIN_AIRPORT")
    else:
        train_df = ensure_weather_columns(train_df)
        test_df = ensure_weather_columns(test_df)

    train_df = add_time_features(train_df)
    test_df = add_time_features(test_df)
    feature_store = build_feature_store(train_df)

    X_train, feature_columns, numeric_columns, categorical_columns = prepare_feature_matrix(
        train_df,
        feature_store=feature_store,
        include_weather=enable_weather,
    )
    X_test, _, _, _ = prepare_feature_matrix(
        test_df,
        feature_store=feature_store,
        include_weather=enable_weather,
    )
    y_train = train_df["IS_DELAYED"]
    y_test = test_df["IS_DELAYED"]

    use_smote = len(X_train) <= 100000
    if not use_smote:
        print("Skipping SMOTE because the sampled training set is large enough and dense resampling would be memory-heavy.")

    print(f"Model type: {model_type}")
    pipeline, parameter_grid = build_pipeline(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        use_smote=use_smote,
        model_type=model_type,
    )
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_grid,
        n_iter=6,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    print("Training model...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_prob = best_model.predict_proba(X_test)[:, 1]
    best_threshold, best_f1 = choose_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= best_threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(best_f1),
        "threshold": float(best_threshold),
    }

    predictor = FlightDelayPredictor(
        model=best_model,
        feature_store=feature_store,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        threshold=best_threshold,
        use_weather_features=enable_weather,
    )
    joblib.dump(predictor, model_output_path)

    print("Best params:", search.best_params_)
    print("Best threshold:", best_threshold)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nModel saved: {model_output_path}")

    return predictor, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train a flight delay prediction model.")
    parser.add_argument("--csv-path", default="flights.csv", help="Path to the training CSV file.")
    parser.add_argument("--model-output", default=MODEL_OUTPUT_PATH, help="Where to save the trained model.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Read only the first N rows to keep local training practical. Use 0 for the full file.",
    )
    parser.add_argument(
        "--enable-weather",
        action="store_true",
        help="Enable weather features. This is only valid if your training data has matching historical weather context.",
    )
    parser.add_argument(
        "--model-type",
        choices=["randomforest", "xgboost"],
        default="randomforest",
        help="Use RandomForest by default for local stability, or opt into XGBoost explicitly.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    max_rows = None if args.max_rows == 0 else args.max_rows
    train_model(
        csv_path=args.csv_path,
        model_output_path=args.model_output,
        max_rows=max_rows,
        enable_weather=args.enable_weather,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()