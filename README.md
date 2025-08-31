# RegressionMadeSimple

A simple and elegant wrapper around scikit-learn for regression tasks. Version 2.0.0 provides a clean, minimal API for common regression models with built-in data splitting capabilities.

## Features

- **Simple API**: One-liner function calls for quick regression fitting
- **Clean Architecture**: Well-organized package structure with clear separation of concerns
- **Sklearn Integration**: Built on top of scikit-learn for reliability and performance
- **Multiple Models**: Support for Linear, Quadratic, and Cubic regression
- **Data Splitting**: Built-in train-test split functionality with customizable ratios
- **Utility Functions**: Built-in validation, I/O, and model management tools

## Installation

```bash
pip install regressionmadesimple
```

## Quick Start

### Function-Style API (One-liner)

```python
import numpy as np
from regressionmadesimple import fit

# Generate sample data
X = np.random.randn(100, 2)
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1

# Fit a model without data splitting
model, results = fit(X, y, model="Linear")
print(f"R² Score: {results['r2_score']:.3f}")

# Fit a model with 80-20 train-test split
model, results = fit(X, y, model="Linear", split_ratio=[8, 2])
print(f"Train R²: {results['train_r2_score']:.3f}")
print(f"Test R²: {results['test_r2_score']:.3f}")

# Access test data from the model
print(f"Test data shape: {model.X_test.shape}")
print(f"Training data shape: {model.X_train.shape}")

# Try different models with splitting
quad_model, quad_results = fit(X, y, model="Quadratic", split_ratio=[7, 3])
cubic_model, cubic_results = fit(X, y, model="Cubic", split_ratio=[8, 2], random_state=42)
```

### Class-Style API

```python
from regressionmadesimple import Linear, Quadratic, Cubic

# Linear regression without splitting
linear = Linear()
linear.fit(X, y)
predictions = linear.predict(X)
score = linear.score(X, y)

# Linear regression with automatic data splitting
linear_split = Linear(split_ratio=[8, 2], random_state=42)
linear_split.fit(X, y)

# Access training and test scores
print(f"Train R²: {linear_split.get_train_score():.3f}")
print(f"Test R²: {linear_split.get_test_score():.3f}")

# Access split data
print(f"Training samples: {len(linear_split.y_train)}")
print(f"Test samples: {len(linear_split.y_test)}")

# Quadratic regression with splitting
quadratic = Quadratic(split_ratio=[7, 3], random_state=123)
quadratic.fit(X, y)
print(quadratic.get_params())

# Cubic regression with splitting
cubic = Cubic(split_ratio=[9, 1])
cubic.fit(X, y)
feature_names = cubic.get_feature_names(['x1', 'x2'])
```

## Data Splitting Feature

The `split_ratio` parameter allows you to automatically split your data into training and testing sets:

```python
# Different ways to specify split ratios
split_ratio = [8, 2]    # 80% train, 20% test
split_ratio = [7, 3]    # 70% train, 30% test
split_ratio = [0.8, 0.2]  # Same as [8, 2] but with decimals

# Using with function API
model, results = fit(X, y, model="Linear", split_ratio=[8, 2], random_state=42)

# Results include both train and test metrics
print(f"Training R²: {results['train_r2_score']:.3f}")
print(f"Test R²: {results['test_r2_score']:.3f}")
print(f"Training samples: {results['n_samples_train']}")
print(f"Test samples: {results['n_samples_test']}")

# Using with class API
model = Linear(split_ratio=[8, 2], random_state=42)
model.fit(X, y)

# Access split data directly
X_train = model.X_train
y_train = model.y_train
X_test = model.X_test
y_test = model.y_test
```

## Package Structure

```
regressionmadesimple/
├── __init__.py                 # Clean, minimal public surface
├── api/                        # One-liner, function-style API
│   ├── __init__.py
│   └── fit.py                  # fit(X, y, model="Linear", split_ratio=[8,2])
├── models/                     # Simple wrappers around sklearn
│   ├── __init__.py
│   ├── linear.py               # Linear regression with splitting
│   ├── quadratic.py            # Quadratic regression with splitting
│   └── cubic.py                # Cubic regression with splitting
└── utils/                      # Utility functions
    ├── __init__.py
    ├── validation.py           # Input validation and data splitting
    └── io.py                   # Model I/O operations
```

## API Reference

### Function API

#### `fit(X, y, model="Linear", split_ratio=None, random_state=None, **kwargs)`

Fit a regression model with a simple one-liner API.

**Parameters:**
- `X`: Training features (array-like)
- `y`: Training targets (array-like)  
- `model`: Model type ("Linear", "Quadratic", "Cubic")
- `split_ratio`: Split ratio as [train_ratio, test_ratio], e.g., [8, 2]
- `random_state`: Random state for reproducible splits
- `**kwargs`: Additional model parameters

**Returns:**
- `fitted_model`: The fitted model instance with test data as attributes if split_ratio provided
- `results`: Dictionary with fit results and metrics

### Model Classes

#### `Linear(fit_intercept=True, normalize=False, split_ratio=None, random_state=None)`

Simple wrapper around scikit-learn's LinearRegression with data splitting.

#### `Quadratic(fit_intercept=True, include_bias=True, split_ratio=None, random_state=None)`

Quadratic (polynomial degree 2) regression using PolynomialFeatures with data splitting.

#### `Cubic(fit_intercept=True, include_bias=True, split_ratio=None, random_state=None)`

Cubic (polynomial degree 3) regression using PolynomialFeatures with data splitting.

**Common Methods:**
- `fit(X, y)`: Fit the model (automatically splits data if split_ratio provided)
- `predict(X)`: Make predictions
- `score(X, y)`: Calculate R² score
- `get_train_score()`: Get R² score on training data
- `get_test_score()`: Get R² score on test data (if available)
- `get_params()`: Get model parameters including split information

**Data Attributes (when split_ratio is used):**
- `X_train`: Training features
- `y_train`: Training targets
- `X_test`: Test features
- `y_test`: Test targets

### Utility Functions

#### Validation and Data Splitting
- `validate_input(X, y)`: Validate and convert input data
- `validate_split_ratio(split_ratio)`: Validate and normalize split ratio
- `split_data(X, y, split_ratio, random_state=None)`: Split data into train/test sets
- `validate_model_params(**params)`: Validate model parameters

#### I/O Operations
- `save_model(model, filepath)`: Save fitted model to disk
- `load_model(filepath)`: Load model from disk
- `export_results(results, filepath, format='json')`: Export results
- `load_data(filepath, **kwargs)`: Load data from various formats
- `save_predictions(predictions, filepath, X=None, feature_names=None)`: Save predictions

## Examples

### Basic Usage with Data Splitting

```python
import numpy as np
from regressionmadesimple import fit, Linear

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1

# Method 1: Function API with splitting
model, results = fit(X, y, model="Linear", split_ratio=[8, 2], random_state=42)
print(f"Train R²: {results['train_r2_score']:.3f}")
print(f"Test R²: {results['test_r2_score']:.3f}")
print(f"Split ratio: {results['split_ratio']}")

# Method 2: Class API with splitting
linear = Linear(split_ratio=[8, 2], random_state=42)
linear.fit(X, y)
print(f"Model: {linear}")
print(f"Train score: {linear.get_train_score():.3f}")
print(f"Test score: {linear.get_test_score():.3f}")
```

### Model Comparison with Splitting

```python
from regressionmadesimple import fit

models = ["Linear", "Quadratic", "Cubic"]
results = {}

for model_type in models:
    model, result = fit(X, y, model=model_type, split_ratio=[8, 2], random_state=42)
    results[model_type] = {
        'train_r2': result['train_r2_score'],
        'test_r2': result['test_r2_score']
    }
    print(f"{model_type}:")
    print(f"  Train R² = {result['train_r2_score']:.3f}")
    print(f"  Test R²  = {result['test_r2_score']:.3f}")

# Find best model based on test performance
best_model = max(results, key=lambda x: results[x]['test_r2'])
print(f"Best model (test R²): {best_model}")
```

### Working with Split Data

```python
from regressionmadesimple import Quadratic

# Create model with data splitting
model = Quadratic(split_ratio=[7, 3], random_state=123)
model.fit(X, y)

# Access split data
print(f"Original data shape: {X.shape}")
print(f"Training data shape: {model.X_train.shape}")
print(f"Test data shape: {model.X_test.shape}")

# Make predictions on both sets
train_predictions = model.predict(model.X_train)
test_predictions = model.predict(model.X_test)

# Calculate custom metrics
from sklearn.metrics import mean_squared_error
train_mse = mean_squared_error(model.y_train, train_predictions)
test_mse = mean_squared_error(model.y_test, test_predictions)

print(f"Train MSE: {train_mse:.3f}")
print(f"Test MSE: {test_mse:.3f}")
```

### Save and Load Models with Split Data

```python
from regressionmadesimple import Linear
from regressionmadesimple.utils import save_model, load_model

# Fit and save model with split data
model = Linear(split_ratio=[8, 2], random_state=42)
model.fit(X, y)
save_model(model, "my_split_model.pkl")

# Load and verify split data is preserved
loaded_model = load_model("my_split_model.pkl")
print(f"Split ratio preserved: {loaded_model.split_ratio}")
print(f"Test data available: {loaded_model.X_test is not None}")
print(f"Test score: {loaded_model.get_test_score():.3f}")
```

## Requirements

- Python >= 3.7
- numpy >= 1.19.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Repository

GitHub: https://github.com/Unknownuserfrommars/regressionmadesimple

## Changelog

### Version 2.0.0
- Initial release of RegressionMadeSimple v2
- Clean, minimal API design
- Support for Linear, Quadratic, and Cubic regression
- **NEW**: Built-in data splitting with `split_ratio` parameter
- **NEW**: Automatic train-test split functionality
- **NEW**: Easy access to split data via model attributes
- Comprehensive utility functions
- Full scikit-learn integration
- Updated repository URL