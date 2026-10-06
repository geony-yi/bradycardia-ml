"""Feature selection and five-fold development of the reported tree models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier


RANDOM_SEED = 42
N_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--id-column", default="case_id")
    parser.add_argument("--held-out-cases", type=int, default=166,
                        help="Total size of the outcome-balanced internal held-out cohort.")
    parser.add_argument("--random-search-iterations", type=int, default=30)
    parser.add_argument("--correlation-threshold", type=float, default=0.90)
    parser.add_argument("--vif-threshold", type=float, default=10.0)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def model_specs(seed: int) -> dict[str, tuple[object, dict[str, list[object]]]]:
    """Prespecified random-search spaces for the five reported classifiers."""
    return {
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=1),
            {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [None, 8, 16, 24],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=seed),
            {
                "model__n_estimators": [100, 300, 500],
                "model__learning_rate": [0.01, 0.05, 0.10],
                "model__max_depth": [2, 3, 4],
                "model__min_samples_leaf": [1, 2, 4],
                "model__subsample": [0.7, 0.85, 1.0],
            },
        ),
        "lightgbm": (
            LGBMClassifier(random_state=seed, n_jobs=1, verbosity=-1),
            {
                "model__n_estimators": [100, 300, 500],
                "model__learning_rate": [0.01, 0.05, 0.10],
                "model__num_leaves": [15, 31, 63],
                "model__max_depth": [-1, 5, 10],
                "model__min_child_samples": [10, 20, 40],
                "model__subsample": [0.7, 0.85, 1.0],
            },
        ),
        "extra_trees": (
            ExtraTreesClassifier(random_state=seed, n_jobs=1),
            {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [None, 8, 16, 24],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        ),
        "xgboost": (
            XGBClassifier(
                random_state=seed, n_jobs=1, eval_metric="logloss", tree_method="hist"
            ),
            {
                "model__n_estimators": [100, 300, 500],
                "model__learning_rate": [0.01, 0.05, 0.10],
                "model__max_depth": [3, 5, 7],
                "model__min_child_weight": [1, 3, 5],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
            },
        ),
    }


def balanced_holdout_indices(y: pd.Series, total_cases: int, seed: int) -> np.ndarray:
    """Select an outcome-balanced held-out cohort without changing development prevalence."""
    if set(pd.unique(y.dropna())) != {0, 1}:
        raise ValueError("The target column must contain both binary classes 0 and 1.")
    if total_cases < 2 or total_cases % 2:
        raise ValueError("--held-out-cases must be an even number of at least two.")
    per_class = total_cases // 2
    rng = np.random.default_rng(seed)
    selections = []
    for label in (0, 1):
        candidates = np.flatnonzero(y.to_numpy() == label)
        if len(candidates) < per_class:
            raise ValueError("A class has fewer cases than required for the balanced hold-out cohort.")
        selections.append(rng.choice(candidates, size=per_class, replace=False))
    return np.sort(np.concatenate(selections))


def prepare_feature_matrix(data: pd.DataFrame, target_column: str, id_column: str) -> pd.DataFrame:
    excluded = {target_column, id_column}
    features = data[[column for column in data.columns if column not in excluded]].copy()
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    all_missing = features.columns[features.isna().all()].tolist()
    return features.drop(columns=all_missing)


def normality_aware_screen(X: pd.DataFrame, y: pd.Series, alpha: float = 0.05) -> tuple[list[str], pd.DataFrame]:
    """Univariate P-value screening using t-test or Mann-Whitney U after Shapiro-Wilk."""
    rows = []
    selected = []
    for column in X.columns:
        positive = X.loc[y == 1, column].dropna().to_numpy(dtype=float)
        negative = X.loc[y == 0, column].dropna().to_numpy(dtype=float)
        if len(positive) < 3 or len(negative) < 3:
            rows.append({"feature": column, "test": "insufficient_data", "p_value": np.nan})
            continue
        # Shapiro-Wilk is defined for up to 5,000 values; deterministic truncation is sufficient here.
        positive_check = positive[:5000]
        negative_check = negative[:5000]
        normal = stats.shapiro(positive_check).pvalue > alpha and stats.shapiro(negative_check).pvalue > alpha
        if normal:
            result = stats.ttest_ind(positive, negative, equal_var=False, nan_policy="omit")
            test = "independent_t_test"
        else:
            result = stats.mannwhitneyu(positive, negative, alternative="two-sided")
            test = "mann_whitney_u"
        p_value = float(result.pvalue)
        rows.append({"feature": column, "test": test, "p_value": p_value})
        if np.isfinite(p_value) and p_value < alpha:
            selected.append(column)
    if not selected:
        raise RuntimeError("No feature passed the univariate P < 0.05 screen.")
    return selected, pd.DataFrame(rows)


def correlation_filter(X: pd.DataFrame, p_values: pd.Series, threshold: float) -> tuple[list[str], list[str]]:
    """Use Spearman correlation to retain the more strongly univariately associated feature."""
    correlation = X.corr(method="spearman").abs()
    remove: set[str] = set()
    columns = list(X.columns)
    for left_index, left in enumerate(columns):
        if left in remove:
            continue
        for right in columns[left_index + 1:]:
            if right in remove or correlation.loc[left, right] < threshold:
                continue
            left_p = p_values.get(left, np.inf)
            right_p = p_values.get(right, np.inf)
            if (right_p, right) < (left_p, left):
                remove.add(left)
                break
            remove.add(right)
    kept = [column for column in columns if column not in remove]
    return kept, sorted(remove)


def vif_filter(X: pd.DataFrame, threshold: float) -> tuple[list[str], list[dict[str, float]]]:
    """Iteratively remove the feature with the largest VIF above the chosen threshold."""
    if X.shape[1] <= 1:
        return list(X.columns), []
    values = SimpleImputer(strategy="most_frequent").fit_transform(X)
    remaining = list(X.columns)
    removed: list[dict[str, float]] = []
    while len(remaining) > 1:
        matrix = pd.DataFrame(values, columns=X.columns)[remaining]
        try:
            design = np.column_stack([np.ones(len(matrix)), matrix.to_numpy()])
            vifs = [
                variance_inflation_factor(design, index + 1)
                for index in range(matrix.shape[1])
            ]
        except Exception:
            vifs = [np.inf] * len(remaining)
        worst_index = int(np.nanargmax(np.nan_to_num(vifs, nan=np.inf, posinf=np.inf)))
        worst_vif = float(vifs[worst_index])
        if np.isfinite(worst_vif) and worst_vif <= threshold:
            break
        removed.append({"feature": remaining[worst_index], "vif": worst_vif})
        remaining.pop(worst_index)
    return remaining, removed


def select_features(X: pd.DataFrame, y: pd.Series, correlation_threshold: float,
                    vif_threshold: float, seed: int) -> tuple[list[str], dict[str, object]]:
    screened, screen_table = normality_aware_screen(X, y)
    p_values = screen_table.set_index("feature")["p_value"]
    correlated_kept, correlated_removed = correlation_filter(
        X[screened], p_values, correlation_threshold
    )
    vif_kept, vif_removed = vif_filter(X[correlated_kept], vif_threshold)
    if len(vif_kept) == 1:
        return vif_kept, {
            "univariate": screen_table.to_dict(orient="records"),
            "correlation_removed": correlated_removed,
            "vif_removed": vif_removed,
            "rfecv_selected": vif_kept,
        }

    imputed = SimpleImputer(strategy="most_frequent").fit_transform(X[vif_kept])
    selector = RFECV(
        estimator=RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
        step=1,
        cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed),
        scoring="roc_auc",
        n_jobs=-1,
    )
    selector.fit(imputed, y)
    selected = list(np.asarray(vif_kept)[selector.support_])
    return selected, {
        "univariate": screen_table.to_dict(orient="records"),
        "correlation_removed": correlated_removed,
        "vif_removed": vif_removed,
        "rfecv_selected": selected,
    }


def build_pipeline(estimator: object, seed: int) -> Pipeline:
    # Each component is fit only inside the current training fold. The scaler gives
    # external data the training-fold location/scale without refitting it externally.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
        ("oversampler", RandomOverSampler(random_state=seed)),
        ("model", estimator),
    ])


def fit_model(X_train: pd.DataFrame, y_train: pd.Series, estimator: object,
              search_space: dict[str, list[object]], iterations: int, seed: int) -> Pipeline:
    pipeline = build_pipeline(estimator, seed)
    inner_cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=search_space,
        n_iter=min(iterations, int(np.prod([len(values) for values in search_space.values()]))),
        scoring="roc_auc",
        cv=inner_cv,
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def prediction_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (probabilities >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def youden_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def recall_target_threshold(y_true: np.ndarray, probabilities: np.ndarray, target_recall: float) -> float:
    positive = np.sort(np.asarray(probabilities)[np.asarray(y_true) == 1])
    if len(positive) == 0:
        raise ValueError("Threshold selection needs at least one positive validation case.")
    required = int(np.ceil(target_recall * len(positive)))
    return float(positive[len(positive) - required])


def prediction_rows(case_ids: pd.Series, y_true: np.ndarray, probabilities: np.ndarray,
                    model: str, fold: int, cohort: str) -> pd.DataFrame:
    return pd.DataFrame({
        "case_id": case_ids.to_numpy(),
        "target": y_true,
        "probability": probabilities,
        "model": model,
        "fold": fold,
        "cohort": cohort,
    })


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data)
    if args.target_column not in data or args.id_column not in data:
        raise ValueError("Input must contain the configured case ID and target columns.")
    if data[args.id_column].isna().any() or data[args.id_column].duplicated().any():
        raise ValueError("The case ID column must contain one non-missing identifier per procedure.")
    y = pd.to_numeric(data[args.target_column], errors="raise").astype(int)
    X = prepare_feature_matrix(data, args.target_column, args.id_column)
    held_out = balanced_holdout_indices(y, args.held_out_cases, args.random_seed)
    development_mask = np.ones(len(data), dtype=bool)
    development_mask[held_out] = False
    dev_X, dev_y = X.loc[development_mask].reset_index(drop=True), y.loc[development_mask].reset_index(drop=True)
    dev_ids = data.loc[development_mask, args.id_column].reset_index(drop=True)
    test_X, test_y = X.loc[~development_mask].reset_index(drop=True), y.loc[~development_mask].reset_index(drop=True)
    test_ids = data.loc[~development_mask, args.id_column].reset_index(drop=True)

    args.output.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.output / "model_artifacts"
    artifact_dir.mkdir(exist_ok=True)
    specs = model_specs(args.random_seed)
    validation_predictions: list[pd.DataFrame] = []
    holdout_predictions: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []

    # Feature selection is performed once using the complete development set.
    # The internal held-out cohort remains untouched, and every CV fold uses the
    # same fixed feature set selected here.
    selected_features, selection_report = select_features(
        dev_X,
        dev_y,
        args.correlation_threshold,
        args.vif_threshold,
        args.random_seed,
    )
    selection_report["selection_scope"] = "complete_development_set"
    selection_report["development_case_count"] = int(len(dev_X))
    selection_reports: dict[str, object] = {
        "development_set": selection_report,
    }

    outer_cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=args.random_seed)
    for fold, (train_index, valid_index) in enumerate(outer_cv.split(dev_X, dev_y)):
        X_train, y_train = dev_X.iloc[train_index], dev_y.iloc[train_index]
        X_valid, y_valid = dev_X.iloc[valid_index], dev_y.iloc[valid_index]
        fold_probabilities: dict[str, np.ndarray] = {}

        for model_name, (estimator, search_space) in specs.items():
            fitted = fit_model(
                X_train[selected_features], y_train, estimator, search_space,
                args.random_search_iterations, args.random_seed + fold,
            )
            validation_probability = fitted.predict_proba(X_valid[selected_features])[:, 1]
            holdout_probability = fitted.predict_proba(test_X[selected_features])[:, 1]
            fold_probabilities[model_name] = validation_probability
            joblib.dump(
                {"model": fitted, "selected_features": selected_features, "model_name": model_name, "fold": fold},
                artifact_dir / f"{model_name}_fold_{fold}.joblib",
            )
            validation_predictions.append(prediction_rows(
                dev_ids.iloc[valid_index], y_valid.to_numpy(), validation_probability, model_name, fold, "validation"
            ))
            holdout_predictions.append(prediction_rows(
                test_ids, test_y.to_numpy(), holdout_probability, model_name, fold, "internal_holdout"
            ))

        ensemble_probability = np.mean(np.column_stack(list(fold_probabilities.values())), axis=1)
        ensemble_holdout = np.mean(np.column_stack([
            holdout_predictions[-len(specs) + position].probability.to_numpy()
            for position in range(len(specs))
        ]), axis=1)
        validation_predictions.append(prediction_rows(
            dev_ids.iloc[valid_index], y_valid.to_numpy(), ensemble_probability, "ensemble", fold, "validation"
        ))
        holdout_predictions.append(prediction_rows(
            test_ids, test_y.to_numpy(), ensemble_holdout, "ensemble", fold, "internal_holdout"
        ))
        for model_name, probability in {**fold_probabilities, "ensemble": ensemble_probability}.items():
            for operating_point, threshold in {
                "youden_j": youden_threshold(y_valid.to_numpy(), probability),
                "recall_0.90": recall_target_threshold(y_valid.to_numpy(), probability, 0.90),
                "recall_0.95": recall_target_threshold(y_valid.to_numpy(), probability, 0.95),
                "recall_1.00": recall_target_threshold(y_valid.to_numpy(), probability, 1.00),
            }.items():
                threshold_rows.append({
                    "model": model_name, "fold": fold, "operating_point": operating_point, "threshold": threshold,
                    **prediction_metrics(y_valid.to_numpy(), probability, threshold),
                })

    pd.concat(validation_predictions, ignore_index=True).to_csv(args.output / "validation_predictions.csv", index=False)
    pd.concat(holdout_predictions, ignore_index=True).to_csv(args.output / "internal_holdout_predictions.csv", index=False)
    data.loc[~development_mask].reset_index(drop=True).to_csv(
        args.output / "internal_holdout_features.csv", index=False
    )
    pd.DataFrame(threshold_rows).to_csv(args.output / "validation_thresholds.csv", index=False)
    (args.output / "feature_selection.json").write_text(json.dumps(selection_reports, indent=2), encoding="utf-8")
    (args.output / "feature_schema.json").write_text(json.dumps(list(X.columns), indent=2), encoding="utf-8")
    print(f"Saved five fold-specific pipelines for {len(specs)} learners in {artifact_dir}")
    print("Internal held-out predictions and validation-derived thresholds were written to the output folder.")


if __name__ == "__main__":
    main()
