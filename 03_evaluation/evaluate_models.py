"""Evaluate fixed fold-specific models without refitting on an evaluation cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--id-column", default="case_id")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--shap-max-samples", type=int, default=500)
    return parser.parse_args()


def prepare_features(data: pd.DataFrame, target_column: str, id_column: str) -> pd.DataFrame:
    X = data.drop(columns=[target_column, id_column]).copy()
    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    return X


def load_bundles(directory: Path) -> list[dict[str, object]]:
    bundles = [joblib.load(path) for path in sorted(directory.glob("*_fold_*.joblib"))]
    if not bundles:
        raise FileNotFoundError("No fold-specific model artifacts were found.")
    needed = {"model", "selected_features", "model_name", "fold"}
    if any(not needed.issubset(bundle) for bundle in bundles):
        raise ValueError("One or more model artifacts do not match the expected release format.")
    return bundles


def metric_row(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (probabilities >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def predict_all(data: pd.DataFrame, X: pd.DataFrame, bundles: list[dict[str, object]],
                target_column: str, id_column: str) -> pd.DataFrame:
    rows = []
    y_true = data[target_column].to_numpy(dtype=int)
    case_ids = data[id_column].to_numpy()
    for bundle in bundles:
        selected = list(bundle["selected_features"])
        probability = bundle["model"].predict_proba(X.reindex(columns=selected))[:, 1]
        rows.append(pd.DataFrame({
            "case_id": case_ids,
            "target": y_true,
            "probability": probability,
            "model": bundle["model_name"],
            "fold": int(bundle["fold"]),
        }))
    base_predictions = pd.concat(rows, ignore_index=True)
    ensemble = (
        base_predictions.groupby(["case_id", "target", "fold"], as_index=False)["probability"]
        .mean()
        .assign(model="ensemble")
    )
    return pd.concat([base_predictions, ensemble], ignore_index=True)


def evaluation_by_threshold(predictions: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "fold", "operating_point", "threshold"}
    if not required.issubset(thresholds.columns):
        raise ValueError(f"Threshold file is missing: {sorted(required.difference(thresholds.columns))}")
    merged = predictions.merge(
        thresholds[list(required)], on=["model", "fold"], how="inner", validate="many_to_many"
    )
    rows = []
    for (model, fold, operating_point, threshold), group in merged.groupby(
        ["model", "fold", "operating_point", "threshold"], sort=True
    ):
        rows.append({
            "model": model,
            "fold": int(fold),
            "operating_point": operating_point,
            "threshold": float(threshold),
            **metric_row(group.target.to_numpy(), group.probability.to_numpy(), float(threshold)),
        })
    return pd.DataFrame(rows)


def build_score_matrices(predictions: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray, list[int]]:
    matrices: dict[str, np.ndarray] = {}
    folds = sorted(predictions.fold.unique())
    reference_case_order = None
    y_true = None
    for model, group in predictions.groupby("model"):
        pivot = group.pivot(index="case_id", columns="fold", values="probability").reindex(columns=folds)
        labels = group.drop_duplicates("case_id").set_index("case_id")["target"].reindex(pivot.index).to_numpy(dtype=int)
        if reference_case_order is None:
            reference_case_order, y_true = pivot.index, labels
        else:
            pivot = pivot.reindex(reference_case_order)
            labels = group.drop_duplicates("case_id").set_index("case_id")["target"].reindex(reference_case_order).to_numpy(dtype=int)
            if not np.array_equal(labels, y_true):
                raise ValueError("Target labels differ between model prediction rows.")
        matrices[str(model)] = pivot.to_numpy(dtype=float).T
    if y_true is None:
        raise RuntimeError("No prediction matrices were created.")
    return matrices, y_true, folds


def threshold_matrix(thresholds: pd.DataFrame, model: str, operating_point: str,
                     folds: list[int]) -> np.ndarray:
    selected = thresholds[
        (thresholds.model == model) & (thresholds.operating_point == operating_point)
    ].set_index("fold")["threshold"]
    missing = set(folds).difference(selected.index)
    if missing:
        raise ValueError(f"Thresholds are missing {model} folds {sorted(missing)} for {operating_point}.")
    return selected.reindex(folds).to_numpy(dtype=float)


def stratified_bootstrap_summary(matrices: dict[str, np.ndarray], y_true: np.ndarray,
                                 thresholds: pd.DataFrame, folds: list[int],
                                 iterations: int, seed: int) -> pd.DataFrame:
    """Use identical case-level stratified resamples for every model and fold."""
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Bootstrap confidence intervals require both outcome classes.")
    points: dict[tuple[str, str], dict[str, float]] = {}
    samples: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    operating_points = sorted(thresholds.operating_point.unique())
    metric_names = ["auroc", "accuracy", "precision", "recall", "f1"]
    for model, score_matrix in matrices.items():
        for operating_point in operating_points:
            try:
                thresholds_by_fold = threshold_matrix(thresholds, model, operating_point, folds)
            except ValueError:
                continue
            point = {
                metric: float(np.mean([
                    metric_row(y_true, score_matrix[index], thresholds_by_fold[index])[metric]
                    for index in range(len(folds))
                ]))
                for metric in metric_names
            }
            points[(model, operating_point)] = point
            samples[(model, operating_point)] = {
                metric: np.empty(iterations, dtype=float) for metric in metric_names
            }

    for iteration in range(iterations):
        index = np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])
        for (model, operating_point), storage in samples.items():
            scores = matrices[model]
            limits = threshold_matrix(thresholds, model, operating_point, folds)
            fold_metrics = [
                metric_row(y_true[index], scores[fold_index, index], limits[fold_index])
                for fold_index in range(len(folds))
            ]
            for metric in storage:
                storage[metric][iteration] = np.mean([row[metric] for row in fold_metrics])

    rows = []
    for key, point in points.items():
        for metric, estimate in point.items():
            values = samples[key][metric]
            rows.append({
                "model": key[0], "operating_point": key[1], "metric": metric,
                "estimate": estimate,
                "ci_95_lower": float(np.quantile(values, 0.025)),
                "ci_95_upper": float(np.quantile(values, 0.975)),
                "bootstrap_iterations": iterations,
            })
    return pd.DataFrame(rows)


def midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return ranks


def delong_auc_covariance(scores: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong covariance for one or more correlated ROC curves."""
    positive, negative = scores[:, :positive_count], scores[:, positive_count:]
    m, n = positive.shape[1], negative.shape[1]
    tx = np.vstack([midrank(row) for row in positive])
    ty = np.vstack([midrank(row) for row in negative])
    tz = np.vstack([midrank(row) for row in scores])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    covariance = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, covariance


def paired_delong(y_true: np.ndarray, reference: np.ndarray, comparison: np.ndarray) -> dict[str, float]:
    order = np.argsort(-y_true)
    positive_count = int(y_true.sum())
    aucs, covariance = delong_auc_covariance(
        np.vstack([reference, comparison])[:, order], positive_count
    )
    difference = float(aucs[0] - aucs[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    standard_error = float(np.sqrt(max(variance, 0.0)))
    z_value = 0.0 if standard_error == 0 and difference == 0 else difference / standard_error
    return {
        "reference_auroc": float(aucs[0]), "comparison_auroc": float(aucs[1]),
        "difference": difference, "standard_error": standard_error,
        "p_value": float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else 0.0,
    }


def delong_table(matrices: dict[str, np.ndarray], y_true: np.ndarray) -> pd.DataFrame:
    reference = matrices["ensemble"].mean(axis=0)
    rows = []
    for model, scores in matrices.items():
        if model == "ensemble":
            continue
        rows.append({"reference": "ensemble", "comparison": model,
                     **paired_delong(y_true, reference, scores.mean(axis=0))})
    return pd.DataFrame(rows)


def calibration_summary(matrices: dict[str, np.ndarray], y_true: np.ndarray, output: Path) -> pd.DataFrame:
    rows = []
    plt.figure(figsize=(5, 5))
    for model, scores in matrices.items():
        probabilities = np.clip(scores.mean(axis=0), 1e-6, 1 - 1e-6)
        design = sm.add_constant(np.log(probabilities / (1 - probabilities)))
        try:
            fitted = sm.Logit(y_true, design).fit(disp=False)
            intercept = float(fitted.params[0])
            slope = float(fitted.params[1])
        except Exception:
            # A finite calibration estimate is not available under perfect separation.
            intercept, slope = np.nan, np.nan
        rows.append({
            "model": model,
            "calibration_in_the_large": intercept,
            "calibration_slope": slope,
            "brier_score": float(brier_score_loss(y_true, probabilities)),
        })
        observed, predicted = calibration_curve(y_true, probabilities, n_bins=5, strategy="quantile")
        plt.plot(predicted, observed, marker="o", label=model)
    plt.plot([0, 1], [0, 1], linestyle=":", color="black", label="ideal")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed proportion")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output / "calibration_plot.png", dpi=200)
    plt.close()
    return pd.DataFrame(rows)


def roc_plot(matrices: dict[str, np.ndarray], y_true: np.ndarray, output: Path) -> None:
    plt.figure(figsize=(6, 5))
    for model, scores in matrices.items():
        probability = scores.mean(axis=0)
        fpr, tpr, _ = roc_curve(y_true, probability)
        plt.plot(fpr, tpr, label=f"{model} (AUROC={roc_auc_score(y_true, probability):.3f})")
    plt.plot([0, 1], [0, 1], linestyle=":", color="black")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output / "roc_curves.png", dpi=200)
    plt.close()


def positive_class_shap_values(values: object) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values[-1])
    array = np.asarray(values.values if hasattr(values, "values") else values)
    return array[..., 1] if array.ndim == 3 and array.shape[-1] == 2 else array


def shap_summary(bundles: list[dict[str, object]], X: pd.DataFrame, output: Path,
                 max_samples: int) -> pd.DataFrame:
    """TreeSHAP importance, aggregated equally across learners and folds on probability scale."""
    try:
        import shap
    except ImportError:
        return pd.DataFrame()
    rows = []
    sample = X.iloc[:max_samples]
    for bundle in bundles:
        selected = list(bundle["selected_features"])
        pipeline = bundle["model"]
        raw = sample.reindex(columns=selected)
        imputed = pd.DataFrame(
            pipeline.named_steps["imputer"].transform(raw), columns=selected, index=raw.index
        )
        transformed = pd.DataFrame(
            pipeline.named_steps["scaler"].transform(imputed), columns=selected, index=raw.index
        )
        estimator = pipeline.named_steps["model"]
        try:
            background = transformed[: min(100, len(transformed))]
            explainer = shap.TreeExplainer(
                estimator, data=background, feature_perturbation="interventional", model_output="probability"
            )
            values = positive_class_shap_values(explainer.shap_values(transformed))
        except Exception:
            # Do not substitute a different explainability method or scale silently.
            continue
        mean_absolute = np.abs(values).mean(axis=0)
        rows.extend({
            "model": bundle["model_name"], "fold": int(bundle["fold"]),
            "feature": feature, "mean_absolute_shap": float(importance),
        } for feature, importance in zip(selected, mean_absolute))
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    detail.to_csv(output / "shap_importance_by_model_fold.csv", index=False)
    ensemble = (
        detail.groupby("feature", as_index=False)["mean_absolute_shap"].mean()
        .sort_values("mean_absolute_shap", ascending=False)
    )
    ensemble.to_csv(output / "shap_importance_ensemble.csv", index=False)
    top = ensemble.head(20).iloc[::-1]
    plt.figure(figsize=(7, max(3, len(top) * 0.25)))
    plt.barh(top.feature, top.mean_absolute_shap)
    plt.xlabel("Mean absolute SHAP value (probability scale)")
    plt.tight_layout()
    plt.savefig(output / "shap_importance_ensemble.png", dpi=200)
    plt.close()
    return detail


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data)
    if args.target_column not in data or args.id_column not in data:
        raise ValueError("Evaluation data must contain the configured case ID and binary target columns.")
    if data[args.id_column].isna().any() or data[args.id_column].duplicated().any():
        raise ValueError("The case ID column must contain one non-missing identifier per procedure.")
    y_true = pd.to_numeric(data[args.target_column], errors="raise").astype(int).to_numpy()
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Evaluation requires both binary outcome classes.")
    args.output.mkdir(parents=True, exist_ok=True)
    X = prepare_features(data, args.target_column, args.id_column)
    bundles = load_bundles(args.artifacts)
    thresholds = pd.read_csv(args.thresholds)
    predictions = predict_all(data, X, bundles, args.target_column, args.id_column)
    predictions.to_csv(args.output / "fold_predictions.csv", index=False)

    performance = evaluation_by_threshold(predictions, thresholds)
    performance.to_csv(args.output / "performance_by_fold.csv", index=False)
    matrices, labels, folds = build_score_matrices(predictions)
    bootstrap = stratified_bootstrap_summary(
        matrices, labels, thresholds, folds, args.bootstrap_iterations, args.random_seed
    )
    bootstrap.to_csv(args.output / "bootstrap_performance.csv", index=False)
    delong_table(matrices, labels).to_csv(args.output / "delong_ensemble_reference.csv", index=False)
    calibration_summary(matrices, labels, args.output).to_csv(args.output / "calibration_summary.csv", index=False)
    roc_plot(matrices, labels, args.output)
    shap_summary(bundles, X, args.output, args.shap_max_samples)
    print(f"Evaluated {len(data)} cases using fixed models; results are in {args.output}")


if __name__ == "__main__":
    main()
