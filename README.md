# RegressionMadeSimple (RMS)

[![PyPI](https://img.shields.io/pypi/v/regressionmadesimple?style=flat-square)](https://pypi.org/project/regressionmadesimple/)
[![Downloads](https://img.shields.io/pypi/dm/regressionmadesimple?style=flat-square)](https://pypi.org/project/regressionmadesimple/)
[![License](https://img.shields.io/github/license/Unknownuserfrommars/regressionmadesimple?style=flat-square)](https://github.com/Unknownuserfrommars/regressionmadesimple/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/regressionmadesimple?style=flat-square)](https://pypi.org/project/regressionmadesimple/)

> A minimalist machine learning wrapper for lazy/skilled devs who want to skip the boilerplate. Built as a clean backdoor to `scikit-learn`.

---

## 🚀 Quickstart

```python
import regressionmadesimple as rms

# Load dataset
df = rms.Preworks.readcsv("./your_data.csv")

# Train a linear regression model
model = rms.Linear(dataset=df, colX="feature", colY="target")

# Predict new values
predicted = model.predict([[5.2], [3.3]])

# Plot prediction vs. test
model.plot_predict([[5.2], [3.3]], predicted).show()
```

---

## 📦 Features

- 🧠 Wraps `sklearn`'s most-used regression model(s) in a friendly API
- 📊 Built-in Plotly visualizations -- planned later support for matplotlib (? on this)
- 🔬 Designed for quick prototyping and educational use
- 🧰 Utility functions via `preworks`:
  - `readcsv(path)` — load CSV
  - `create_random_dataset(...)` — create random datasets (for demos)
- One-liner regression setup
- `.summary()` and `.plot()` for quick insight
- Global config system: `rms.options`
- Accepts pre-split data (`X_train`, `y_train`, etc.)
- Easily extendable — logistic, trees, etc. coming soon
- MIT Licensed

---

## Project LINK
https://unknownuserfrommars.github.io/regressionmadesimple/
> PS: Changelog also can be accessed from there

---

## ✅ Installation

```bash
pip install regressionmadesimple
```

> Or install the dev version:

```bash
git clone https://github.com/Unknownuserfrommars/regressionmadesimple.git
cd regressionmadesimple
pip install -e .
```

---

## 📁 Project Structure

```text
regressionmadesimple/
├── __init__.py
├── base_class.py
├── linear.py
├── logistic.py        # (soon)
├── tree.py            # (soon)
├── utils_preworks.py
```

---

## 🧪 Tests
Coming soon under a `/tests` folder using `pytest`

---

## 📜 License
[MIT License](./LICENSE)

---

## 🛠 Author
Made with ❤️ by [Unknownuserfrommars](github.com/Unknownuserfrommars)
:)

---

## 🌌 Ideas for Future Versions

- `Logistic()` and `DecisionTree()` models
- `.summary()` for *all* models
- Export/save models
- Visual explainability (feature importance, SHAP)

---

## ⭐ Star this project if you like lazy ML.