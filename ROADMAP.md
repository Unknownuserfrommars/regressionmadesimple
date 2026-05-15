# 🗺️ RegressionMadeSimple — Roadmap

> **Current version:** 4.0.0  
> **Status:** Stable single-feature regression toolkit  
> **Inspired by:** `MLWorkflow` — Kevin's experiment-based ML framework

---

## 📋 Version Naming Convention

| Series | Status | Description |
|--------|--------|-------------|
| **v4.x** | Current | Regression + Experiment Workflow + model comparison |
| **v5.x** | Future | Multi-feature regression + tuning |
| **v6.x** | Future | Classification support |
| **v7.x** | Future | Clustering + auto-ML + production toolkit |

Each major version bumps systematically. Features may shift between minor versions
depending on development pace and real-world priorities.

---

## 🏗️ v4.x Series — From Thin Wrapper to Full Experiment Framework

The key insight from `MLWorkflow` is that a **thin sklearn wrapper** is fine for quick fits,
but the real value comes from an **experiment management layer** — multi-split comparisons,
model ranking, deduplication, and iterative feature engineering.

### v4.1.0 — The Experiment Workflow

The centerpiece: a new `rms.Experiment` class inspired by `MLWorkflow`.

- [ ] **`rms.Experiment(df, target, out_path)`** — stateful workflow that owns the full pipeline
- [ ] **Smart column typing** — auto-detect bool / numeric / categorical columns, one-hot encode cats
  - 0/1 numeric columns automatically treated as boolean
- [ ] **Multi-split management** — store multiple `train_splits` with metadata (random_state, sizes)
  - Each split records `X_train`, `y_train`, `X_test`, `y_test`, `X_validation`, `y_validation`
- [ ] **Built-in validation split** — `train_test_split(test_size=0.2, validation_size=0.1)` → 70/20/10
- [ ] **Scaler integration** — fit `StandardScaler` / `MinMaxScaler` on train, auto-transform test/val
- [ ] **`fit_models(model_dict)`** — accept dict of `{stem: [estimator, estimator, ...]}` like MLWorkflow
  - Each model gets an auto-incremented key (e.g., `knn_0`, `knn_1`)
- [ ] **Per-model metadata** — store scalar, train_split key, feature columns, timestamp alongside predictions
- [ ] **Full state persistence via dill** — `experiment.save(path)` / `Experiment.load(path)` — save everything
- [ ] **Legacy code removal** — delete `curves.py`, `base_class.py` — they're dead code since v3

### v4.2.0 — Model Comparison & Ranking

- [ ] **`experiment.get_model_ranking(metric="r2_score", top_n=10)`** — ranked DataFrame, respects `_LOWER_IS_BETTER`
- [ ] **`experiment.view_overall_results()`** — pivot table: models × datasets with all metrics
- [ ] **`experiment.view_overfitting()`** — train vs test score gap per model, flag overfit models
- [ ] **`experiment.view_stem_comparison(stem="knn")`** — compare hyperparameters within a model family
- [ ] **`experiment.view_feature_comparison()`** — group by feature column sets used
- [ ] **`_LOWER_IS_BETTER` frozenset** — "accuracy" → sort descending, "mse" → sort ascending, etc.
- [ ] **Model deduplication** — `_build_model_metadata()` + `_is_duplicate()` → skip re-fitting identical configs
  - Recursively normalizes sklearn params (numpy types, nested dicts, estimators) into comparable Python

### v4.3.0 — Iterative Feature Engineering & Rich Metrics

- [ ] **`experiment.add_columns(new_columns_df)`** — propagate new features to **all existing splits** retroactively
  - This enables the "tweak a feature, re-evaluate all models" loop
- [ ] **Rich metric collection** — per split (train / test / val), compute full metric dict:
  - Regression: `r2_score`, `mse`, `mae`, `rmse`, `max_error`, `MAPE`, `explained_variance`, `median_absolute_error`, `mean_squared_log_error`
  - Automatic skipping of MSLE when targets contain negatives
- [ ] **Residual plots** — Plotly scatter with hover data, reference line at y=0
- [ ] **Adjusted R²** metric (penalizes extra features)
- [ ] **`experiment.view_model_key("knn_0")`** — drill into a single model's full results
- [ ] **`experiment.get_best_estimator(metric="r2_score")`** — return the best fitted model object
- [ ] **CustomCurve stabilization** — move from "experimental" to stable
  - More basis functions: `tan(x)`, `abs(x)`, `1/x`, `sigmoid(x)`
  - User-defined callable basis functions: `lambda x: ...`
- [ ] **Better error messages** everywhere (wrong columns, missing data, invalid scaler names)

---

## 🚀 v5.x Series — Regression 2.0

### v5.0.0 — Multi-Feature Regression

- [ ] **Multi-feature support** — `colX` accepts list of column names: `colX=['x1', 'x2', 'x3']`
- [ ] **Function API** (`api.fit`) updated to accept multi-feature X
- [ ] **Feature importance / coefficient ranking** output in summary
- [ ] **Interaction terms** — `rms.models.Linear(data, ['x1', 'x2'], 'y', interactions=True)`
- [ ] **Regularization variants** — Ridge, Lasso, ElasticNet
- [ ] **Expand Experiment** to handle multi-feature column selection like MLWorkflow's `columns` parameter
  - `columns="all"` (auto: num + bool + dummies) or `columns=["x1", "x2"]` (explicit)

### v5.1.0 — Hyperparameter Tuning

- [ ] **Grid search helper** — light wrapper around `sklearn.model_selection.GridSearchCV`
- [ ] **Random search helper** — light wrapper around `sklearn.model_selection.RandomizedSearchCV`
- [ ] **Integration with Experiment** — `experiment.grid_search(model_stem, estimator, param_grid)` stores all candidates
- [ ] **Auto-pick best** from search results with dedup protection

### v5.2.0 — Cross-Validation

- [ ] **k-Fold CV** — built-in cross-validation scores when no explicit test split
- [ ] **Stratified k-Fold** for classification (prep for v6)
- [ ] **AIC / BIC** for model comparison
- [ ] **Confidence intervals** on coefficients

---

## 🎯 v6.x Series — Classification & Beyond

### v6.0.0 — Classification Support

- [ ] **Binary classification models**:
  - `rms.models.Logistic` (wraps `LogisticRegression`)
  - `rms.models.KNN` (wraps `KNeighborsClassifier`)
  - `rms.models.DecisionTree`
  - `rms.models.RandomForest`
- [ ] **Multi-class classification**
- [ ] **Classification metrics** — accuracy, precision, recall, F1, confusion matrix plotting
  - Weighted / macro / micro averages
- [ ] **ROC / AUC** curve plotting (binary + multi-class OVR)
- [ ] **Probability metrics** — log_loss, brier_score (when `predict_proba` is available)
- [ ] **Specificity** calculation (MLWorkflow-style, from confusion matrix)
- [ ] **Expand Experiment** — `is_classifier` awareness, classification-specific view methods

### v6.1.0 — Clustering & Dimensionality Reduction

- [ ] **Clustering models** — KMeans, DBSCAN (`rms.clustering.*`)
- [ ] **PCA wrapper** — `rms.preprocessing.PCA`
- [ ] **t-SNE / UMAP** visualization helpers
- [ ] **Silhouette score, inertia, Davies-Bouldin** clustering metrics

---

## 🧪 v7.x Series — Auto-ML & Production

### v7.0.0 — Auto-ML

- [ ] **Auto-model selection** — try multiple model families, rank by score, pick best
- [ ] **Feature selection helpers** — correlation-based, variance threshold, mutual information
- [ ] **Feature engineering** — date parsing, binning, polynomial expansion, interaction auto-detect
- [ ] **Pipeline composability** — `rms.Pipeline([("scaler", MinMaxScaler()), ("model", Linear)]`

### v7.1.0 — Production Toolkit

- [ ] **ONNX export** — deploy models outside Python
- [ ] **Model versioning** — save metadata alongside models (sklearn version, feature names, timestamp)
- [ ] **Batch prediction** — optimized for large datasets with progress tracking
- [ ] **Progress bars** for long-running fits / searches
- [ ] **Experiment report export** — auto-generate HTML summary of all model results

---

## 📚 Documentation (Continual)

- [ ] **API reference docs** — auto-generated from docstrings
- [ ] **Jupyter notebook examples** — one notebook per feature area
- [ ] **Experiment workflow tutorial** — step-by-step from data → fit → compare → export
- [ ] **Migration guide** v4.x → v5.x
- [ ] **Google Colab quick-start badge** in README
- [ ] **CI/CD** — GitHub Actions for automated testing on PRs

---

## 🔗 Design Philosophy (from MLWorkflow)

This roadmap is heavily influenced by Kevin's `MLWorkflow` design. Core principles:

| Principle | Why |
|---|---|
| **Stateful Experiment > stateless wrapper** | Compare models, re-run on new features, inspect everything |
| **Auto-detect column types** | Less boilerplate, catch misconfigurations early |
| **Model metadata + dedup** | Never re-fit the same config twice by accident |
| **Metric-aware sorting** | Lower-is-better metrics (MSE) vs higher-is-better (R²) handled transparently |
| **Multi-split management** | One dataset → many split strategies → compare robustness |
| **Full state persistence** | `dill` the whole experiment → pick up where you left off |
| **Iterative feature engineering** | `add_columns` retroactively propagates to all splits — game changer |

---

## ⚠️ Notes

This roadmap is aspirational and will evolve based on feedback and usage.
If you'd like to prioritize something, open a [GitHub Discussion](https://github.com/Unknownuserfrommars/regressionmadesimple/discussions).
