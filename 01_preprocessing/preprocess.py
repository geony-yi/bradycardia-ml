"""Create a de-identified modelling table from physiological recordings.

The script implements the published five-minute pre-administration feature window.
It intentionally accepts all recording paths and track names as runtime inputs rather
than embedding any local study paths or device-specific identifiers.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SUMMARY_FEATURES = (
    "mean", "median", "std", "rms", "skewness", "kurtosis",
    "min", "max", "q1", "q3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path,
                        help="Plain-text list: one physiological track per line.")
    parser.add_argument("--outcome-track", required=True,
                        help="Heart-rate track used to derive the binary outcome.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-id-column", default="case_id")
    parser.add_argument("--record-column", default="record_path")
    parser.add_argument("--administration-column", default="administration_time")
    parser.add_argument("--age-column", default="age")
    parser.add_argument("--sex-column", default="sex")
    parser.add_argument("--height-column", default="height_cm")
    parser.add_argument("--weight-column", default="weight_kg")
    parser.add_argument("--sampling-rate", type=int, default=500)
    parser.add_argument("--window-minutes", type=float, default=5.0)
    parser.add_argument("--horizon-minutes", type=float, default=0.0,
                        help="Window end in minutes before administration (0, 1, 2, or 3).")
    parser.add_argument("--min-track-coverage", type=float, default=0.50)
    parser.add_argument("--coverage-column", default=None,
                        help="Optional metadata column identifying the development cohort.")
    parser.add_argument("--coverage-value", default="1")
    return parser.parse_args()


def read_track_list(path: Path) -> list[str]:
    tracks = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    tracks = [track for track in tracks if track and not track.startswith("#")]
    if not tracks:
        raise ValueError("The track list is empty.")
    return tracks


def numeric_signal(frame: pd.DataFrame, requested_track: str) -> tuple[pd.Series, pd.Series]:
    """Return timestamps and the requested numeric signal from a VitalDB frame."""
    if "Time" not in frame.columns:
        raise ValueError("Recording reader did not return a Time column.")
    candidate_columns = [requested_track] if requested_track in frame.columns else []
    candidate_columns += [
        column for column in frame.columns
        if column not in {"Time", "Timestamp"} and column not in candidate_columns
    ]
    if not candidate_columns:
        raise ValueError(f"No signal column returned for track: {requested_track}")
    values = pd.to_numeric(frame[candidate_columns[0]], errors="coerce")
    times = pd.to_datetime(frame["Time"], errors="coerce")
    return times, values


def administration_index(times: pd.Series, administration_time: object) -> int:
    """Locate a timestamp, accepting a full timestamp or an HH:MM:SS recording time."""
    raw = str(administration_time).strip()
    if not raw:
        raise ValueError("Administration time is missing.")

    if len(raw) <= 8 and raw.count(":") >= 1:
        matches = np.flatnonzero(times.dt.strftime("%H:%M:%S").eq(raw).to_numpy())
        if len(matches):
            return int(matches[-1])

    target = pd.to_datetime(raw, errors="coerce")
    if pd.isna(target):
        raise ValueError(f"Cannot parse administration time: {raw}")
    offsets = np.abs((times - target).dt.total_seconds().to_numpy(dtype=float))
    if not np.isfinite(offsets).any():
        raise ValueError("No valid timestamps were available.")
    return int(np.nanargmin(offsets))


def signal_window(values: pd.Series, administration_idx: int, sampling_rate: int,
                  window_minutes: float, horizon_minutes: float) -> np.ndarray:
    end = administration_idx - round(horizon_minutes * 60 * sampling_rate)
    start = end - round(window_minutes * 60 * sampling_rate)
    if start < 0 or end <= start:
        raise ValueError("The requested feature window is not fully available.")
    return values.iloc[start:end].to_numpy(dtype=float)


def summary_features(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {name: np.nan for name in SUMMARY_FEATURES}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            "mean": float(np.nanmean(finite)),
            "median": float(np.nanmedian(finite)),
            "std": float(np.nanstd(finite)),
            "rms": float(np.sqrt(np.nanmean(finite ** 2))),
            "skewness": float(stats.skew(finite, nan_policy="omit")),
            "kurtosis": float(stats.kurtosis(finite, nan_policy="omit")),
            "min": float(np.nanmin(finite)),
            "max": float(np.nanmax(finite)),
            "q1": float(np.nanquantile(finite, 0.25)),
            "q3": float(np.nanquantile(finite, 0.75)),
        }


def variability_features(values: np.ndarray, sampling_rate: int, kind: str) -> dict[str, float]:
    """Calculate HRV from ECG or PRV from photoplethysmography when applicable."""
    finite = np.asarray(values, dtype=float)
    if np.isfinite(finite).mean() < 0.95:
        return {}
    try:
        import neurokit2 as nk
        if kind == "ecg":
            cleaned = nk.ecg_clean(finite, sampling_rate=sampling_rate)
            _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
            output = nk.hrv(info["ECG_R_Peaks"], sampling_rate=sampling_rate, show=False)
            prefix = "hrv"
        elif kind == "ppg":
            cleaned = nk.ppg_clean(finite, sampling_rate=sampling_rate)
            _, info = nk.ppg_peaks(cleaned, sampling_rate=sampling_rate)
            output = nk.hrv(info["PPG_Peaks"], sampling_rate=sampling_rate, show=False)
            prefix = "prv"
        else:
            return {}
    except Exception:
        return {}

    return {
        f"{prefix}_{column.removeprefix('HRV_').lower()}": float(value)
        for column, value in output.iloc[0].items()
        if np.isscalar(value) and np.isfinite(value)
    }


def outcome_from_heart_rate(values: pd.Series, administration_idx: int,
                            sampling_rate: int) -> int:
    """Return 1 for a >=20% decrease 30 seconds to 10 minutes after administration."""
    baseline = float(values.iloc[administration_idx])
    lower = administration_idx + 30 * sampling_rate
    upper = administration_idx + 10 * 60 * sampling_rate
    post = values.iloc[lower:upper].to_numpy(dtype=float)
    if not np.isfinite(baseline) or not np.isfinite(post).any():
        raise ValueError("Heart-rate data are unavailable for outcome derivation.")
    return int(np.nanmin(post) <= 0.80 * baseline)


def sex_to_binary(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"m", "male", "1"}:
        return 1.0
    if text in {"f", "female", "0"}:
        return 0.0
    return np.nan


def load_recording(record_path: object, track: str, sampling_rate: int):
    try:
        import vitaldb
    except ImportError as error:
        raise ImportError("Install vitaldb before running the preprocessing stage.") from error
    record = vitaldb.VitalFile(str(record_path))
    frame = record.to_pandas(
        track, 1 / sampling_rate, return_datetime=True, return_timestamp=True
    )
    return numeric_signal(frame, track)


def require_preprocessing_dependencies() -> None:
    try:
        import neurokit2  # noqa: F401
        import vitaldb  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Install the preprocessing dependencies in requirements.txt before running this stage."
        ) from error


def append_demographics(row: pd.Series, args: argparse.Namespace) -> dict[str, object]:
    height = pd.to_numeric(pd.Series([row.get(args.height_column)]), errors="coerce").iloc[0]
    weight = pd.to_numeric(pd.Series([row.get(args.weight_column)]), errors="coerce").iloc[0]
    bmi = np.nan if pd.isna(height) or pd.isna(weight) or height <= 0 else weight / (height / 100) ** 2
    return {
        "case_id": row[args.case_id_column],
        "age": row.get(args.age_column),
        "sex": sex_to_binary(row.get(args.sex_column)),
        "height_cm": height,
        "weight_kg": weight,
        "bmi": bmi,
    }


def drop_low_coverage_tracks(features: pd.DataFrame, coverage_mask: pd.Series,
                             minimum_coverage: float) -> pd.DataFrame:
    track_columns: dict[str, list[str]] = {}
    for column in features.columns:
        if "__" in column:
            track, _ = column.split("__", 1)
            track_columns.setdefault(track, []).append(column)
    remove: list[str] = []
    for columns in track_columns.values():
        coverage = features.loc[coverage_mask, columns].notna().any(axis=1).mean()
        if coverage < minimum_coverage:
            remove.extend(columns)
    return features.drop(columns=remove)


def main() -> None:
    args = parse_args()
    require_preprocessing_dependencies()
    if args.sampling_rate != 500:
        raise ValueError("The reported analysis uses a 500-Hz waveform sampling rate.")
    if args.horizon_minutes not in {0.0, 1.0, 2.0, 3.0}:
        raise ValueError("Horizon must be 0, 1, 2, or 3 minutes.")
    metadata = pd.read_csv(args.metadata)
    required = {args.case_id_column, args.record_column, args.administration_column}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    tracks = read_track_list(args.tracks)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for _, metadata_row in metadata.iterrows():
        case = append_demographics(metadata_row, args)
        try:
            times, heart_rate = load_recording(
                metadata_row[args.record_column], args.outcome_track, args.sampling_rate
            )
            dose_idx = administration_index(times, metadata_row[args.administration_column])
            case["target"] = outcome_from_heart_rate(heart_rate, dose_idx, args.sampling_rate)

            for track in tracks:
                try:
                    track_times, values = load_recording(
                        metadata_row[args.record_column], track, args.sampling_rate
                    )
                    track_idx = administration_index(track_times, metadata_row[args.administration_column])
                    window = signal_window(
                        values, track_idx, args.sampling_rate,
                        args.window_minutes, args.horizon_minutes,
                    )
                    for name, value in summary_features(window).items():
                        case[f"{track}__{name}"] = value
                    upper = track.upper()
                    kind = "ecg" if "ECG" in upper else "ppg" if "PLETH" in upper else ""
                    for name, value in variability_features(window, args.sampling_rate, kind).items():
                        case[f"{track}__{name}"] = value
                except Exception:
                    # A missing individual track remains missing; the case is retained.
                    continue
            rows.append(case)
        except Exception as error:
            failures.append({"case_id": str(case["case_id"]), "reason": str(error)})

    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError("No records produced a complete outcome and feature window.")
    if args.coverage_column and args.coverage_column in metadata.columns:
        allowed = metadata.loc[metadata[args.coverage_column].astype(str) == str(args.coverage_value), args.case_id_column]
        coverage_mask = features["case_id"].isin(allowed)
    else:
        coverage_mask = pd.Series(True, index=features.index)
    features = drop_low_coverage_tracks(features, coverage_mask, args.min_track_coverage)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(args.output.with_name("excluded_cases.csv"), index=False)
    print(f"Wrote {len(features)} procedures and {features.shape[1]} columns to {args.output}")
    print(f"Excluded {len(failures)} procedures with incomplete outcome or feature windows.")


if __name__ == "__main__":
    main()
