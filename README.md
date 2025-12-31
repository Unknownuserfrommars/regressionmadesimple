# RegressionMadeSimple v3.0.0 🚀

A minimalist machine learning toolkit that wraps `scikit-learn` for quick prototyping. Just `import regressionmadesimple as rms` and go!

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/regressionmadesimple?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=Total+Downloads)](https://pepy.tech/projects/regressionmadesimple)

## What's New in v3.0.0 🎉

RegressionMadeSimple v3.0.0 brings major improvements to make your regression workflow even simpler and more powerful:

### 🏗️ Model Registry Pattern
- **New API**: Access models via `rms.models.Linear`, `rms.models.Quadratic`, `rms.models.Cubic`
- **Better Organization**: All models now live in a dedicated `models` submodule
- **Type Safety**: Use class-based model specification for better IDE support

### 🔧 Enhanced Base Class
- **Common Functionality**: Eliminated code duplication across model classes
- **Model Serialization**: Save and load trained models with `save_model()` and `load_model()`
- **Additional Metrics**: New scoring methods including `r2_score()`, `mae()`, `rmse()`
- **Shared Logic**: Unified train/test splitting and plotting functionality

### 🔄 Backward Compatibility
- **Legacy Support**: Old API still works (with deprecation warnings)
- **Smooth Migration**: Gradual transition path to new API
- **No Breaking Changes**: Existing code continues to work

---

## Installation

```bash
pip install regressionmadesimple
```

Or install from source:

```bash
git clone https://github.com/Unknownuserfrommars/regressionmadesimple.git
cd regressionmadesimple
pip install -e .
```

---

## Quick Start

### Basic Usage (v3.0.0 API - Recommended)

```python
import regressionmadesimple as rms
import pandas as pd

# Load your data
data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'y': [2.1, 4.2, 6.1, 8.3, 10.2, 12.1, 14.3, 16.2, 18.1, 20.3]
})

# Fit a linear regression model (new API)
model = rms.models.Linear(data, 'x', 'y')

# Make predictions
predictions = model.predict([[11], [12]])
print(predictions)

# Get model summary with metrics
summary = model.summary()
print(f"R² Score: {summary['r2_score']:.4f}")
print(f"RMSE: {summary['rmse']:.4f}")
print(f"MAE: {summary['mae']:.4f}")

# Save the model
model.save_model('my_linear_model.pkl')

# Load it later
loaded_model = rms.models.BaseModel.load_model('my_linear_model.pkl')
```

### Using the Wrapper (Both APIs Supported)

```python
# New v3.0.0 API (recommended)
model = rms.LinearRegressionModel.fit(
    data, 'x', 'y', 
    model=rms.models.Quadratic,
    testsize=0.2
)

# Legacy API (deprecated, shows warning)
model = rms.LinearRegressionModel.fit(
    data, 'x', 'y', 
    model='quadratic',  # String-based (deprecated)
    testsize=0.2
)
```

---

## Migration Guide: v2.x → v3.0.0

### What Changed?

1. **Model Access**: Models now accessible via `rms.models.*`
2. **Enhanced Features**: New serialization and scoring methods
3. **Deprecation Warnings**: String-based model specification shows warnings

### Migration Examples

#### Before (v2.x)
```python
import regressionmadesimple as rms

# Direct instantiation
model = rms.Linear(data, 'x', 'y')

# Wrapper with string
model = rms.LinearRegressionModel.fit(data, 'x', 'y', model='linear')
```

#### After (v3.0.0 - Recommended)
```python
import regressionmadesimple as rms

# Direct instantiation (still works!)
model = rms.Linear(data, 'x', 'y')  # Backward compatible

# Or use new model registry (recommended)
model = rms.models.Linear(data, 'x', 'y')

# Wrapper with class
model = rms.LinearRegressionModel.fit(data, 'x', 'y', model=rms.models.Linear)
```

### Migration Checklist

- [ ] Update imports to use `rms.models.*` for new projects
- [ ] Replace string-based model specification with class-based
- [ ] Take advantage of new features: `save_model()`, `r2_score()`, `mae()`, `rmse()`
- [ ] Test existing code (it should still work with deprecation warnings)
- [ ] Plan migration before v4.0.0 (when legacy API will be removed)

---

## Features

### Available Models

| Model | Class | Description |
|-------|-------|-------------|
| **Linear** | `rms.models.Linear` | Simple linear regression (y = mx + b) |
| **Quadratic** | `rms.models.Quadratic` | Polynomial regression (degree=2) |
| **Cubic** | `rms.models.Cubic` | Polynomial regression (degree=3) |
| **Custom Curve** | `rms.models.CustomCurve` | Custom basis functions (experimental) |

### Core Functionality

#### 1. Model Training
```python
# Automatic train/test split (default)
model = rms.models.Linear(data, 'x', 'y', testsize=0.2, randomstate=42)

# Use entire dataset (no split)
model = rms.models.Linear(data, 'x', 'y', train_test_split=False)

# Provide pre-split data
model = rms.models.Linear(
    data, 'x', 'y',
    train_test_split=False,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test
)
```

#### 2. Predictions
```python
# Single prediction
result = model.predict([[5.5]])

# Multiple predictions
results = model.predict([[5.5], [6.0], [6.5]])
```

#### 3. Model Evaluation (New in v3.0.0!)
```python
# Get comprehensive summary
summary = model.summary()
print(summary)
# Output: {
#   'model': 'Linear Regression',
#   'mse': 0.0234,
#   'rmse': 0.1530,
#   'mae': 0.1234,
#   'r2_score': 0.9876,
#   'coef': [[2.0123]],
#   'intercept': [0.0987]
# }

# Individual metrics
print(f"R² Score: {model.r2_score()}")
print(f"Mean Absolute Error: {model.mae()}")
print(f"Root Mean Squared Error: {model.rmse()}")
print(f"Mean Squared Error: {model.mse()}")
```

#### 4. Model Serialization (New in v3.0.0!)
```python
# Save trained model
model.save_model('models/my_regression_model.pkl')

# Load model later
from regressionmadesimple.models import BaseModel
loaded_model = BaseModel.load_model('models/my_regression_model.pkl')

# Use loaded model
predictions = loaded_model.predict(new_data)
```

#### 5. Visualization
```python
# Plot training/test data with predictions
fig = model.plot_y_train()
fig.show()  # For Plotly backend

# Plot predictions on new data
fig = model.plot_predict(X_new, y_new)
fig.show()
```

### Configuration Options

```python
import regressionmadesimple as rms

# View current options
print(rms.options)

# Change plotting backend
rms.options.plot.backend = 'matplotlib'  # or 'plotly' (default)

# Change training defaults
rms.options.training.test_size = 0.25
rms.options.training.random_state = 42

# Save custom options
rms.save_options('my_config.json')

# Load options later
rms.load_options('my_config.json')

# Reset to defaults
rms.reset_options()
```

---

## Advanced Examples

### Example 1: Quadratic Regression with Custom Split

```python
import regressionmadesimple as rms
import pandas as pd
import numpy as np

# Generate sample data with quadratic relationship
x = np.linspace(0, 10, 100)
y = 2*x**2 - 3*x + 5 + np.random.normal(0, 5, 100)
data = pd.DataFrame({'x': x, 'y': y})

# Fit quadratic model
model = rms.models.Quadratic(data, 'x', 'y', testsize=0.3, randomstate=123)

# Evaluate
summary = model.summary()
print(f"Model: {summary['model']}")
print(f"R² Score: {summary['r2_score']:.4f}")
print(f"RMSE: {summary['rmse']:.4f}")

# Visualize
fig = model.plot_y_train()
fig.show()

# Save for later use
model.save_model('quadratic_model.pkl')
```

### Example 2: Custom Curve Fitting (Experimental)

```python
import regressionmadesimple as rms
import pandas as pd
import numpy as np

# Generate data with sinusoidal pattern
x = np.linspace(0, 2*np.pi, 100)
y = 3*np.sin(x) + 2*np.cos(x) + np.random.normal(0, 0.5, 100)
data = pd.DataFrame({'x': x, 'y': y})

# Fit custom curve with trigonometric basis functions
model = rms.models.CustomCurve(
    data, 'x', 'y',
    basis_funcs=['x', 'sin(x)', 'cos(x)', 'x^2']
)

# Evaluate
print(f"Basis Functions: {model.basis_funcs}")
print(f"R² Score: {model.r2_score():.4f}")

# Make predictions
x_new = np.linspace(0, 2*np.pi, 50)
predictions = model.predict(x_new.reshape(-1, 1))
```

### Example 3: Comparing Multiple Models

```python
import regressionmadesimple as rms
import pandas as pd

# Load data
data = pd.read_csv('your_data.csv')

# Try different models
models = {
    'Linear': rms.models.Linear(data, 'x', 'y'),
    'Quadratic': rms.models.Quadratic(data, 'x', 'y'),
    'Cubic': rms.models.Cubic(data, 'x', 'y'),
}

# Compare performance
print("Model Comparison:")
print("-" * 60)
for name, model in models.items():
    summary = model.summary()
    print(f"{name:12} | R²: {summary['r2_score']:.4f} | "
          f"RMSE: {summary['rmse']:.4f} | MAE: {summary['mae']:.4f}")

# Save the best model
best_model = models['Quadratic']  # Based on your analysis
best_model.save_model('best_model.pkl')
```

---

## API Reference

### Model Classes

#### `rms.models.Linear`
Linear regression model (y = mx + b)

**Methods:**
- `__init__(dataset, colX, colY, **kwargs)` - Initialize and fit model
- `predict(X_new)` - Make predictions
- `summary()` - Get model summary with metrics
- `save_model(filepath)` - Save model to disk
- `plot_y_train()` - Plot training/test data
- `plot_predict(X_new, y_new)` - Plot predictions
- `r2_score()` - Calculate R² score
- `mae()` - Calculate Mean Absolute Error
- `rmse()` - Calculate Root Mean Squared Error
- `mse()` - Get Mean Squared Error

#### `rms.models.Quadratic`
Quadratic regression model (polynomial degree=2)

Same methods as Linear.

#### `rms.models.Cubic`
Cubic regression model (polynomial degree=3)

Same methods as Linear.

#### `rms.models.CustomCurve`
Custom curve fitting with user-defined basis functions
Still testing! Some features may not be working!

**Additional Parameters:**
- `basis_funcs` - List of basis functions: `["x", "x^2", "x^3", "cos(x)", "sin(x)", "log(x)", "exp(x)", "sqrt(x)"]`

### Wrapper Class

#### `rms.LinearRegressionModel`

**Methods:**
- `fit(dataset, colX, colY, model, **kwargs)` - Fit a regression model
  - `model` can be a class (e.g., `rms.models.Linear`) or string (deprecated)

---

## Configuration

### Options Object

```python
rms.options.plot.backend          # 'plotly' or 'matplotlib'
rms.options.training.test_size    # Default: 0.15
rms.options.training.random_state # Default: 1
rms.options.linear.auto_split     # Default: True
rms.options.quadratic.auto_split  # Default: True
```

### Functions

- `rms.save_options(filepath)` - Save current options to JSON
- `rms.load_options(filepath)` - Load options from JSON
- `rms.reset_options()` - Reset to default options

---

## Roadmap

### Planned for v3.x
- [ ] Additional models
- [ ] KNN regression support
- [ ] Cross-validation utilities
- [ ] Hyperparameter tuning helpers
- [ ] More visualization options
- [ ] Support for the creation of a validation dataset (apart from training & testing dataset)

### Future (v4.0.0 and later)
- [ ] Remove deprecated string-based API
- [ ] Classification models
- [ ] Pipeline support
- [ ] Feature engineering utilities

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built on top of [scikit-learn](https://scikit-learn.org/)
- Visualization powered by [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/) (Matplotlib support will be added in future versions!)
- Data handling with [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/Unknownuserfrommars/regressionmadesimple/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Unknownuserfrommars/regressionmadesimple/discussions)

---

## Changelog

### v3.0.0 (2025-01-01)
- ✨ **New**: Model registry pattern (`rms.models.*`)
- ✨ **New**: Enhanced BaseModel with common functionality
- ✨ **New**: Model serialization (`save_model()`, `load_model()`)
- ✨ **New**: Additional scoring metrics (`r2_score()`, `mae()`, `rmse()`)
- 🔧 **Improved**: Refactored codebase to eliminate duplication
- 🔧 **Improved**: Better type hints throughout
- ⚠️ **Deprecated**: String-based model specification (still works with warnings)
- 📚 **Docs**: Comprehensive README with migration guide

### v2.0.0 (Previous Release)
- Initial public release
- Linear, Quadratic, Cubic regression models
- Basic plotting and evaluation
- Options configuration system

---


Made with ❤️ by [Unknownuserfrommars](https://github.com/Unknownuserfrommars)
