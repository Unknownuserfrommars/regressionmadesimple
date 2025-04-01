# RegressionMadeSimple (RMS)

> **"A minimalist ML backdoor to sklearn. Just import `rms` and go."**
![PyPI](https://img.shields.io/pypi/v/regressionmadesimple?style=flat-square)
![PyPI - Downloads](https://img.shields.io/pypi/dm/regressionmadesimple?label=Downloads&style=flat-square)
![License](https://img.shields.io/github/license/Unknownuserfrommars/regressionmadesimple?style=flat-square)
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

---

## Project LINK
https://unknownuserfrommars.github.io/regressionmadesimple/
#### PS: Changelog also can be accessed from there. (still actively developing)

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

## 🧠 Author
**Unknownuserfrommars** — built with 💡, ❤️, and `while True:` in VSCode and PyCharm.

###### kind note: ignore that last statement :)

---

## 🌌 Ideas for Future Versions

- `Logistic()` and `DecisionTree()` models
- `.summary()` for **all** models
- `rms.fit(df, target="y", model="linear")` one-liner
- Export/save models
- Visual explainability (feature importance, SHAP)

---

## ⭐ Star this project if you like lazy ML. No boilerplate, just vibes.
### Also: Buy me a coffee (maybe) coming soon.
