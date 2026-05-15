"""
RegressionMadeSimple v4.1.0 — Experiment Workflow

An `Experiment` manages the full ML pipeline:
- Smart column typing (auto-detect bool / numeric / categorical)
- Multi-split management (train / test / validation)
- Scalers
- Fitting multiple models across splits
- Full state persistence via dill

Inspired by Kevin's MLWorkflow.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import dill
import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .options import options as rms_options

# ---------------------------------------------------------------------------
# Metrics where *lower* is better — used by ranking / sorting helpers
# ---------------------------------------------------------------------------
_LOWER_IS_BETTER = frozenset(
    {
        "mean_squared_error",
        "mse",
        "mean_absolute_error",
        "mae",
        "max_error",
        "rmse",
        "log_loss",
        "brier_score",
    }
)

# ---------------------------------------------------------------------------
# Default settings (users can override via Experiment.settings)
# ---------------------------------------------------------------------------
_default_settings = SimpleNamespace(
    random_state=67,
    test_size=0.2,
    validation_size=0.0,
    scaler="minmax",
)


class Experiment:
    """A stateful ML experiment manager.

    Handles data preparation, train/test/validation splitting, model fitting
    with metadata tracking, and full-state persistence via dill.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target : str
        Name of the target column.
    out_path : str or Path
        Directory where experiment artifacts will be saved.
    get_dummies_drop_first : bool, default=False
        Whether to drop the first dummy column when one-hot encoding categories.
    random_state : int, optional
        Global random state. Falls back to ``settings.random_state``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        out_path: Union[str, Path],
        *,
        get_dummies_drop_first: bool = False,
        random_state: Optional[int] = None,
    ):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        self.settings = SimpleNamespace(
            random_state=random_state or _default_settings.random_state,
            test_size=_default_settings.test_size,
            validation_size=_default_settings.validation_size,
            scaler=_default_settings.scaler,
        )

        self.target = target
        self.out_path = Path(out_path)
        self.out_path.mkdir(parents=True, exist_ok=True)

        # Track row identity for reproducible splits
        df = df.copy()
        df["_row_id"] = np.arange(len(df))

        # --- Smart column typing ---
        X = df.drop(columns=[target])
        y = df[[target]]

        self._bool_cols = list(X.select_dtypes(include="bool").columns)

        num_cols_temp = list(X.select_dtypes(include="number").columns)
        self._bool_cols += self._update_bool_cols(num_cols_temp, X)
        self._bool_cols = list(dict.fromkeys(self._bool_cols))  # dedup, preserve order

        self._num_cols = list(
            np.setdiff1d(num_cols_temp, self._bool_cols + ["_row_id"])
        )
        self._cat_cols = list(
            X.select_dtypes(include=["object", "category"]).columns
        )

        # One-hot encode categorical columns
        if self._cat_cols:
            dummies = pd.get_dummies(
                X[self._cat_cols], drop_first=get_dummies_drop_first
            )
            self._dummy_cols = dummies.columns.tolist()
        else:
            dummies = pd.DataFrame(index=X.index)
            self._dummy_cols = []

        # Assemble processed dataframe
        df_parts = [
            X[["_row_id"] + self._num_cols],
            X[self._bool_cols] if self._bool_cols else pd.DataFrame(index=X.index),
            dummies,
            y,
        ]
        self.df = pd.concat(
            [p for p in df_parts if not (isinstance(p, pd.DataFrame) and p.empty)],
            axis=1,
        )

        # Stores for splits and model results
        self.train_splits: Dict[str, dict] = {}
        self.model_results: Dict[str, dict] = {}

        # Create initial split
        self.train_test_split(
            random_state=self.settings.random_state,
            test_size=self.settings.test_size,
            validation_size=self.settings.validation_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_test_split(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_state: Optional[int] = None,
    ):
        """Create a new train/test(/validation) split and store it.

        Parameters
        ----------
        test_size : float, default=0.2
            Fraction of data for testing (between 0 and 0.5).
        validation_size : float, default=0.0
            Fraction of data for validation (between 0 and 0.5).
        random_state : int, optional
            Random state for reproducibility.
        """
        assert 0 < test_size < 0.5, "test_size should be between 0 and 0.5."
        assert 0 <= validation_size < 0.5, (
            "validation_size should be between 0 and 0.5."
        )

        train_size = 1 - test_size - validation_size
        assert 0.5 <= train_size < 1, (
            f"train_size ({train_size:.3f}) should be between 0.5 and 1."
        )
        assert abs(test_size + validation_size + train_size - 1.0) < 1e-6, (
            "test_size + validation_size + train_size must sum to 1."
        )

        rs = random_state if random_state is not None else self.settings.random_state

        feature_cols = [c for c in self.df.columns if c != self.target]
        X_all = self.df[feature_cols]
        y_all = self.df[self.target]

        if validation_size == 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=test_size, random_state=rs
            )
            X_val = y_val = None
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_all,
                y_all,
                test_size=test_size + validation_size,
                random_state=rs,
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=test_size / (test_size + validation_size),
                random_state=rs,
            )

        split_key = self._next_train_split_key()
        self.train_splits[split_key] = {
            "X_train": X_train.reset_index(drop=True),
            "y_train": y_train.reset_index(drop=True),
            "X_test": X_test.reset_index(drop=True),
            "y_test": y_test.reset_index(drop=True),
            "X_validation": (
                X_val.reset_index(drop=True) if X_val is not None else None
            ),
            "y_validation": (
                y_val.reset_index(drop=True) if y_val is not None else None
            ),
            "meta": {
                "random_state": rs,
                "test_size": test_size,
                "train_size": train_size,
                "validation_size": validation_size,
            },
        }

        return split_key

    def fit_models(
        self,
        model_dict: Dict[str, List[Any]],
        scaler: Optional[str] = None,
        train_split: str = "train_0",
        columns: Union[str, List[str]] = "all",
    ):
        """Fit one or more models across the specified split.

        Parameters
        ----------
        model_dict : dict of str -> list of estimators
            Keys are model stems (e.g. ``"knn"``), values are lists of sklearn
            estimator instances (e.g. ``[KNeighborsClassifier(n=3), ...]``).
        scaler : str, optional
            One of ``"standard"`` or ``"minmax"``.  Falls back to
            ``self.settings.scaler``.
        train_split : str, default="train_0"
            Key of the train split to use.
        columns : str or list of str, default="all"
            Which columns to use. ``"all"`` uses auto-detected numeric + bool + dummy
            columns. A list can include original cat column names (expanded to dummies).
        """
        scaler = scaler or self.settings.scaler
        scaler_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
        }
        if scaler not in scaler_map:
            raise ValueError(
                f"scaler must be one of {list(scaler_map.keys())}, got '{scaler}'"
            )

        if train_split not in self.train_splits:
            raise ValueError(
                f"Train split '{train_split}' not found. "
                f"Available splits: {list(self.train_splits.keys())}"
            )

        split_data = self.train_splits[train_split]
        split_meta = split_data.get("meta", {})

        # Resolve columns -------------------------------------------------
        cols: List[str] = []
        bad_cols: List[str] = []

        if columns == "all":
            cols = self._num_cols + self._bool_cols + self._dummy_cols
        elif isinstance(columns, list):
            for c in columns:
                if c in self._num_cols or c in self._bool_cols or c in self._dummy_cols:
                    cols.append(c)
                elif c in self._cat_cols:
                    expanded = [
                        col for col in self._dummy_cols if col.startswith(c + "_")
                    ]
                    cols.extend(expanded)
                else:
                    bad_cols.append(c)
        else:
            raise TypeError(
                f"columns must be 'all' or a list of column names, got {type(columns)}"
            )

        if bad_cols:
            raise ValueError(f"Columns not found: {bad_cols}")

        cols = list(dict.fromkeys(cols))  # dedup preserving order

        # Prepare data slices ---------------------------------------------
        X_train = split_data["X_train"][cols].copy()
        y_train = split_data["y_train"].copy()
        X_test = split_data["X_test"][cols].copy()
        y_test = split_data["y_test"].copy()
        X_val = (
            split_data["X_validation"][cols].copy()
            if split_data.get("X_validation") is not None
            else None
        )
        y_val = (
            split_data["y_validation"].copy()
            if split_data.get("y_validation") is not None
            else None
        )

        # Scale numeric columns --------------------------------------------
        num_cols_used = [c for c in cols if c in self._num_cols]
        _scaler = scaler_map[scaler]
        if num_cols_used:
            _scaled = _scaler.fit_transform(X_train[num_cols_used].astype(float))
            X_train[num_cols_used] = _scaled
            _scaled_test = _scaler.transform(X_test[num_cols_used].astype(float))
            X_test[num_cols_used] = _scaled_test
            if X_val is not None:
                _scaled_val = _scaler.transform(X_val[num_cols_used].astype(float))
                X_val[num_cols_used] = _scaled_val

        # Fit each model ---------------------------------------------------
        for model_stem, model_list in model_dict.items():
            for estimator in model_list:
                model_key = self._next_model_key(model_stem)

                candidate_meta = self._build_model_metadata(
                    model_stem=model_stem,
                    estimator=estimator,
                    scaler=scaler,
                    train_split=train_split,
                    feature_cols=cols,
                )

                # Dedup check
                is_dup, existing_key = self._is_duplicate(candidate_meta)
                if is_dup:
                    print(
                        f"  Skipping duplicate. Config already exists as: {existing_key}"
                    )
                    continue

                # Clone to avoid mutating the user's original estimator
                est = clone(estimator)
                est.fit(X_train, y_train)

                # Predictions
                y_train_pred = self._get_preds(est, X_train)
                y_test_pred = self._get_preds(est, X_test)
                y_val_pred = (
                    self._get_preds(est, X_val) if X_val is not None else None
                )

                # For classifiers: also get class predictions (not proba)
                is_clf = is_classifier(est)
                if is_clf:
                    y_train_cls = est.predict(X_train)
                    y_test_cls = est.predict(X_test)
                    y_val_cls = est.predict(X_val) if X_val is not None else None
                else:
                    y_train_cls = y_train_pred
                    y_test_cls = y_test_pred
                    y_val_cls = y_val_pred

                # Metrics
                metrics_train = self._collect_metrics(
                    est, X_train, y_train, y_train_cls, is_clf, y_pred_proba=y_train_pred if is_clf else None
                )
                metrics_test = self._collect_metrics(
                    est, X_test, y_test, y_test_cls, is_clf, y_pred_proba=y_test_pred if is_clf else None
                )
                metrics_val = {}
                if X_val is not None and y_val is not None and y_val_pred is not None:
                    metrics_val = self._collect_metrics(
                        est, X_val, y_val, y_val_cls, is_clf, y_pred_proba=y_val_pred if is_clf else None
                    )

                self.model_results[model_key] = {
                    "model": est,
                    "model_stem": model_stem,
                    "scaler": scaler,
                    "train_split": train_split,
                    "feature_cols": cols,
                    "fitted_at": datetime.now().isoformat(),
                    "is_classifier": is_clf,
                    "predictions": {
                        "train": y_train_cls,
                        "test": y_test_cls,
                        "validation": y_val_cls,
                    },
                    "predict_probas": {
                        "train": y_train_pred if is_clf else None,
                        "test": y_test_pred if is_clf else None,
                        "validation": y_val_pred if is_clf else None,
                    } if is_clf else None,
                    "metrics": {
                        "train": metrics_train,
                        "test": metrics_test,
                        "validation": metrics_val,
                    },
                    "metadata": candidate_meta,
                }

                metric_val = metrics_test.get(
                    "accuracy",
                    metrics_test.get("r2_score", ["?"]),
                )[0]
                print(
                    f"  + {model_key:20s}"
                    f"  test_{'acc' if is_clf else 'r2'}: {metric_val:.4f}"
                )

        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Union[str, Path]] = None):
        """Save the entire experiment to disk via dill.

        Parameters
        ----------
        path : str or Path, optional
            File path.  Defaults to ``{out_path}/experiment.dill``.
        """
        path = Path(path) if path else self.out_path / "experiment.dill"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            dill.dump(self, f)
        print(f"Experiment saved to {path}")
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> "Experiment":
        """Load a previously saved experiment.

        Parameters
        ----------
        path : str or Path
            Path to a ``.dill`` file created by ``Experiment.save()``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Experiment file not found: {path}")
        with open(path, "rb") as f:
            obj = dill.load(f)
        print(f"Experiment loaded from {path}")
        return obj

    # ------------------------------------------------------------------
    # Column helpers
    # ------------------------------------------------------------------

    def add_columns(self, new_columns: pd.DataFrame):
        """Add new feature columns and propagate them to all existing splits.

        Parameters
        ----------
        new_columns : pd.DataFrame
            New features indexed by original row id (must align with ``self.df``).
        """
        # Align to the original processed DataFrame
        aligned = new_columns.reindex(self.df.index)
        for col in aligned.columns:
            self.df[col] = aligned[col]

        # Detect column type
        for col in aligned.columns:
            if col in self.df.columns:
                if aligned[col].dtype == bool or aligned[col].nunique() <= 2:
                    if col not in self._bool_cols:
                        self._bool_cols.append(col)
                elif aligned[col].dtype in ("object", "category"):
                    if col not in self._cat_cols:
                        self._cat_cols.append(col)
                else:
                    if col not in self._num_cols:
                        self._num_cols.append(col)

        # Propagate to existing splits
        for split_key, split_data in self.train_splits.items():
            for x_key in ("X_train", "X_test", "X_validation"):
                existing = split_data.get(x_key)
                if existing is not None:
                    new_data = aligned.reindex(existing.index)
                    existing[new_data.columns] = new_data[new_data.columns]

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_preds(estimator, X):
        """Get predictions, preferring ``predict_proba`` when available."""
        if hasattr(estimator, "predict_proba"):
            try:
                return estimator.predict_proba(X)
            except Exception:
                pass
        return estimator.predict(X)

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _collect_metrics(self, model, X, y_true, y_pred, is_clf: bool, y_pred_proba=None) -> dict:
        """Compute a dictionary of metric_name -> [value] for one dataset split."""
        import sklearn.metrics as skm

        result: Dict[str, list] = {}

        if is_clf:
            result["accuracy"] = [float(skm.accuracy_score(y_true, y_pred))]
            result["precision_weighted"] = [
                float(
                    skm.precision_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )
                )
            ]
            result["recall_weighted"] = [
                float(
                    skm.recall_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )
                )
            ]
            result["f1_weighted"] = [
                float(
                    skm.f1_score(y_true, y_pred, average="weighted", zero_division=0)
                )
            ]

            # Probabilistic metrics (require predict_proba)
            if y_pred_proba is not None:
                try:
                    result["log_loss"] = [float(skm.log_loss(y_true, y_pred_proba))]
                    unique_classes = np.unique(y_true)
                    if len(unique_classes) == 2:
                        result["roc_auc"] = [
                            float(skm.roc_auc_score(y_true, y_pred_proba[:, 1]))
                        ]
                        result["brier_score"] = [
                            float(skm.brier_score_loss(y_true, y_pred_proba[:, 1]))
                        ]
                    else:
                        result["roc_auc"] = [
                            float(
                                skm.roc_auc_score(
                                    y_true,
                                    y_pred_proba,
                                    multi_class="ovr",
                                    average="weighted",
                                )
                            )
                        ]
                except Exception:
                    pass
        else:
            result["r2_score"] = [float(skm.r2_score(y_true, y_pred))]
            result["mean_squared_error"] = [
                float(skm.mean_squared_error(y_true, y_pred))
            ]
            result["rmse"] = [
                float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
            ]
            result["mean_absolute_error"] = [
                float(skm.mean_absolute_error(y_true, y_pred))
            ]
            result["explained_variance_score"] = [
                float(skm.explained_variance_score(y_true, y_pred))
            ]
            result["max_error"] = [float(skm.max_error(y_true, y_pred))]
            result["MAPE"] = [
                float(skm.mean_absolute_percentage_error(y_true, y_pred))
            ]
            result["median_absolute_error"] = [
                float(skm.median_absolute_error(y_true, y_pred))
            ]
            if np.all(np.array(y_true) >= 0) and np.all(np.array(y_pred) >= 0):
                result["mean_squared_log_error"] = [
                    float(skm.mean_squared_log_error(y_true, y_pred))
                ]

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_bool_cols(self, num_cols: list, data: pd.DataFrame) -> list:
        """Detect numeric columns that contain only 0/1 values."""
        candidates = [c for c in num_cols if data[c].nunique() <= 2]
        result = []
        for col in candidates:
            unique_vals = set(data[col].dropna().unique())
            if unique_vals <= {0, 1}:
                result.append(col)
        return result

    def _next_train_split_key(self) -> str:
        existing = [
            int(k.split("_", 1)[1])
            for k in self.train_splits
            if k.startswith("train_") and k.split("_", 1)[1].isdigit()
        ]
        return f"train_{max(existing) + 1}" if existing else "train_0"

    def _next_model_key(self, model_stem: str) -> str:
        existing = [
            int(k.rsplit("_", 1)[1])
            for k in self.model_results
            if k.startswith(model_stem) and k.rsplit("_", 1)[1].isdigit()
        ]
        return f"{model_stem}_{max(existing) + 1}" if existing else f"{model_stem}_0"

    def _build_model_metadata(
        self,
        model_stem: str,
        estimator,
        scaler: str,
        train_split: str,
        feature_cols: list,
    ) -> dict:
        """Build a normalised metadata dict for dedup / identification."""
        return {
            "model_stem": model_stem,
            "estimator_params": self._normalize_metadata_value(
                estimator.get_params()
            ),
            "estimator_class": type(estimator).__name__,
            "estimator_module": type(estimator).__module__,
            "scaler": scaler,
            "train_split": train_split,
            "feature_cols": sorted(feature_cols),
        }

    @staticmethod
    def _normalize_metadata_value(value):
        """Recursively normalise a value for comparison.

        Converts numpy types, nested dicts, and estimator instances into
        plain-Python comparable representations.
        """
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: Experiment._normalize_metadata_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return sorted(
                [Experiment._normalize_metadata_value(v) for v in value],
                key=str,
            )
        if hasattr(value, "get_params"):
            # sklearn estimator -> capture its params
            return {
                "_type": type(value).__name__,
                "_module": type(value).__module__,
                "params": Experiment._normalize_metadata_value(value.get_params()),
            }
        return str(value)

    def _is_duplicate(self, candidate_meta: dict) -> tuple:
        """Check if a model configuration already exists.

        Returns
        -------
        (is_duplicate, existing_key_or_empty_str)
        """
        for existing_key, existing_result in self.model_results.items():
            if existing_result.get("metadata") == candidate_meta:
                return True, existing_key
        return False, ""
