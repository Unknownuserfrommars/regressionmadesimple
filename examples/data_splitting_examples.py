#!/usr/bin/env python3
"""
Data splitting examples for RegressionMadeSimple.
"""

import numpy as np
import matplotlib.pyplot as plt
from regressionmadesimple import fit, Linear, Quadratic, Cubic
from regressionmadesimple.utils import split_data


def main():
    """Demonstrate data splitting functionality of RegressionMadeSimple."""
    
    print("=== RegressionMadeSimple v2.0.0 Data Splitting Examples ===\n")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    
    # Example 1: Basic data splitting with function API
    print("1. Basic Data Splitting with Function API")
    print("-" * 45)
    
    X = np.random.randn(n_samples, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n_samples) * 0.2
    
    # Without splitting
    model_no_split, results_no_split = fit(X, y, model="Linear")
    print(f"Without splitting - R² Score: {results_no_split['r2_score']:.3f}")
    print(f"Total samples used: {results_no_split['n_samples_train']}")
    
    # With 80-20 split
    model_split, results_split = fit(X, y, model="Linear", split_ratio=[8, 2], random_state=42)
    print(f"With [8,2] split - Train R²: {results_split['train_r2_score']:.3f}")
    print(f"With [8,2] split - Test R²: {results_split['test_r2_score']:.3f}")
    print(f"Training samples: {results_split['n_samples_train']}")
    print(f"Test samples: {results_split['n_samples_test']}")
    print(f"Test data available in model: {model_split.X_test is not None}\n")
    
    # Example 2: Different split ratios
    print("2. Different Split Ratios")
    print("-" * 26)
    
    split_ratios = [[9, 1], [8, 2], [7, 3], [6, 4]]
    
    for ratio in split_ratios:
        model, results = fit(X, y, model="Linear", split_ratio=ratio, random_state=42)
        train_pct = ratio[0] / sum(ratio) * 100
        test_pct = ratio[1] / sum(ratio) * 100
        print(f"Split {ratio} ({train_pct:.0f}%-{test_pct:.0f}%): "
              f"Train R²={results['train_r2_score']:.3f}, "
              f"Test R²={results['test_r2_score']:.3f}")
    print()
    
    # Example 3: Class API with data splitting
    print("3. Class API with Data Splitting")
    print("-" * 33)
    
    # Linear model with splitting
    linear = Linear(split_ratio=[8, 2], random_state=42)
    linear.fit(X, y)
    
    print(f"Linear model: {linear}")
    print(f"Train score: {linear.get_train_score():.3f}")
    print(f"Test score: {linear.get_test_score():.3f}")
    print(f"Training data shape: {linear.X_train.shape}")
    print(f"Test data shape: {linear.X_test.shape}")
    
    # Quadratic model with different split
    quadratic = Quadratic(split_ratio=[7, 3], random_state=123)
    quadratic.fit(X, y)
    
    print(f"\nQuadratic model: {quadratic}")
    print(f"Train score: {quadratic.get_train_score():.3f}")
    print(f"Test score: {quadratic.get_test_score():.3f}")
    
    # Cubic model with splitting
    cubic = Cubic(split_ratio=[9, 1], random_state=456)
    cubic.fit(X, y)
    
    print(f"\nCubic model: {cubic}")
    print(f"Train score: {cubic.get_train_score():.3f}")
    print(f"Test score: {cubic.get_test_score():.3f}\n")
    
    # Example 4: Model comparison with splitting
    print("4. Model Comparison with Data Splitting")
    print("-" * 39)
    
    models = ["Linear", "Quadratic", "Cubic"]
    comparison_results = {}
    
    for model_type in models:
        model, results = fit(X, y, model=model_type, split_ratio=[8, 2], random_state=42)
        comparison_results[model_type] = {
            'train_r2': results['train_r2_score'],
            'test_r2': results['test_r2_score'],
            'model': model
        }
        print(f"{model_type:>10}: Train R²={results['train_r2_score']:.3f}, "
              f"Test R²={results['test_r2_score']:.3f}")
    
    # Find best model based on test performance
    best_model_name = max(comparison_results, key=lambda x: comparison_results[x]['test_r2'])
    best_test_score = comparison_results[best_model_name]['test_r2']
    print(f"Best model (test R²): {best_model_name} ({best_test_score:.3f})\n")
    
    # Example 5: Manual data splitting utility
    print("5. Manual Data Splitting Utility")
    print("-" * 33)
    
    # Use the split_data utility function directly
    X_train, X_test, y_train, y_test = split_data(X, y, [8, 2], random_state=42)
    
    print(f"Original data shape: {X.shape}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Fit model manually on split data
    manual_model = Linear()
    manual_model.fit(X_train, y_train)
    
    train_score = manual_model.score(X_train, y_train)
    test_score = manual_model.score(X_test, y_test)
    
    print(f"Manual split - Train R²: {train_score:.3f}")
    print(f"Manual split - Test R²: {test_score:.3f}\n")
    
    # Example 6: Working with model parameters and split data
    print("6. Model Parameters with Split Data")
    print("-" * 36)
    
    model_with_params = Quadratic(
        fit_intercept=True,
        include_bias=False,
        split_ratio=[7, 3],
        random_state=789
    )
    model_with_params.fit(X, y)
    
    params = model_with_params.get_params()
    print("Model parameters:")
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: array shape {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # Example 7: Decimal split ratios
    print("\n7. Decimal Split Ratios")
    print("-" * 24)
    
    decimal_ratios = [[0.8, 0.2], [0.75, 0.25], [0.9, 0.1]]
    
    for ratio in decimal_ratios:
        model, results = fit(X, y, model="Linear", split_ratio=ratio, random_state=42)
        print(f"Split {ratio}: Train R²={results['train_r2_score']:.3f}, "
              f"Test R²={results['test_r2_score']:.3f}")
    
    print("\n=== Data Splitting Examples completed successfully! ===")


if __name__ == "__main__":
    main()