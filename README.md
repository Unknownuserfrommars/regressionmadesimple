# RegressionMadeSimple v4.1.0 🚀

A minimalist ML toolkit wrapping `scikit-learn` for quick prototyping.
Just `import regressionmadesimple as rms` and go!

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/regressionmadesimple?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=Total+Downloads)](https://pepy.tech/projects/regressionmadesimple)

---

## What's New in v4.1.0 🎉

### 🧪 Experiment Workflow (New)

A stateful `rms.Experiment` class that manages the full ML pipeline — smart column
typing, multi-split management, scaler integration, model deduplication, and full
state persistence via dill.

```python
import regressionmadesimple as rms
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "cat": ["a", "b", "a", "b", "a"],
    "y": [2.1, 4.2, 6.1, 8.3, 10.2]
})

exp = rms.Experiment(df, target="y", out_path="./experiment")
exp.fit_models({
    "lr": [LinearRegression()],
})

# Explore
results = exp.model_results["lr_0"]
print(results["metrics"]["test"]["r2_score"])

# Persist everything
exp.save()
loaded = rms.Experiment.load("./experiment/experiment.dill")
```

**Key features:**
- **Smart column typing** — bool, 0/1-as-bool, numeric, categorical auto-detected
- **Multi-split management** — multiple train/test/validation splits with metadata
- **Scaler integration** — MinMaxScaler (default) or StandardScaler
- **Validation split** — 70/20/10 three-way splits
- **Model dedup** — identical configs auto-skipped
- **Rich metrics** — r², MSE, RMSE, MAE, MAPE, explained variance + classification metrics
- **dill persistence** — save/load full experiment (data + models + metadata)
- **`add_columns()`** — propagate new features to all existing splits retroactively
- **Classifier support** — works with LogisticRegression, KNN, etc.

### 🧹 Cleanup

- Removed legacy `curves.py` and `base_class.py` (dead code since v3)
- Fixed scaler dtype warnings

---

## Installation

```bash
pip install regressionmadesimple
```

Or from source:

```bash
git clone https://github.com/Unknownuserfrommars/regressionmadesimple.git
cd regressionmadesimple
pip install -e .
```

---

## Quick Start

### Experiment API (v4.1.0)

```python
import regressionmadesimple as rms
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load data
data = pd.DataFrame({
    "x1": range(50),
    "x2": __import__("numpy").random.randn(50),
    "cat": __import__("numpy").random.choice(["a", "b", "c"], 50),
    "y": [2*v + __import__("numpy").random.randn()*0.5 for v in range(50)],
})

# Create experiment
exp = rms.Experiment(data, target="y", out_path="./my_exp")

# Fit multiple models at once
exp.fit_models({
    "lr": [LinearRegression()],
    "rf": [RandomForestRegressor(n_estimators=50, random_state=42)],
})

# Compare results
for key, res in exp.model_results.items():
    r2 = res["metrics"]["test"]["r2_score"][0]
    print(f"{key}: R² = {r2:.4f}")

# Save everything
exp.save()
```

### Classic Model API (v4.x)

```python
import regressionmadesimple as rms

data = pd.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [2.1, 4.2, 6.1, 8.3, 10.2]
})

model = rms.models.Linear(data, "x", "y")
print(f"R²: {model.r2_score():.4f}")
print(f"RMSE: {model.rmse():.4f}")

model.save_model("my_model.pkl")
```

---

## Available Models

| Model | Class | Description |
|-------|-------|-------------|
| **Linear** | `rms.models.Linear` | Simple linear regression (y = mx + b) |
| **Quadratic** | `rms.models.Quadratic` | Polynomial regression (degree=2) |
| **Cubic** | `rms.models.Cubic` | Polynomial regression (degree=3) |
| **Custom Curve** | `rms.models.CustomCurve` | Custom basis functions |

---

## Roadmap 🗺️

See [ROADMAP.md](ROADMAP.md) for the full development plan covering v4.x through v7.x.

---

## Contributing

PRs welcome!

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m "Add some AmazingFeature"`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

MIT — see [LICENSE](LICENSE).

---

## Changelog

### v4.1.0-dev (2026-05-15)
- ✨ **New**: `rms.Experiment` — full experiment workflow with smart column typing,
  multi-split management, scaler integration, model dedup, dill persistence
- ✨ **New**: `_LOWER_IS_BETTER` — exported frozenset for smart metric sorting
- 🧹 **Removed**: Legacy `curves.py` and `base_class.py` (dead code)

### v4.0.0 (2025-01-01)
- Model registry pattern (`rms.models.*`)
- Enhanced BaseModel with common functionality
- Model serialization (`save_model()`, `load_model()`)
- Additional scoring metrics (`r2_score()`, `mae()`, `rmse()`)
- String-based model specification removed

### v3.0.0
- Refactored codebase, model registry, deprecation warnings

### v2.0.0
- Initial public release

---

Made with ❤️ by [Unknownuserfrommars](https://github.com/Unknownuserfrommars)
