#!/usr/bin/env python3
"""
Basic usage examples for RegressionMadeSimple.
"""

import numpy as np
import matplotlib.pyplot as plt
from regressionmadesimple import fit, Linear, Quadratic, Cubic


def main():
    """Demonstrate basic usage of RegressionMadeSimple."""
    
    print("=== RegressionMadeSimple v2.0.0 Examples ===\n")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    
    # Example 1: Linear relationship
    print("1. Linear Regression Example")
    print("-" * 30)
    
    X_linear = np.random.randn(n_samples, 2)
    y_linear = 2 * X_linear[:, 0] + 3 * X_linear[:, 1] + np.random.randn(n_samples) * 0.1
    
    # Function API
    model, results = fit(X_linear, y_linear, model="Linear")
    print(f"Function API - R² Score: {results['r2_score']:.3f}")
    print(f"Coefficients: {results['coefficients']}")
    print(f"Intercept: {results['intercept']:.3f}")
    
    # Class API
    linear = Linear()
    linear.fit(X_linear, y_linear)
    print(f"Class API - R² Score: {linear.score(X_linear, y_linear):.3f}")
    print(f"Model: {linear}\n")
    
    # Example 2: Quadratic relationship
    print("2. Quadratic Regression Example")
    print("-" * 32)
    
    X_quad = np.random.randn(n_samples, 1)
    y_quad = 2 * X_quad[:, 0]**2 + X_quad[:, 0] + np.random.randn(n_samples) * 0.2
    
    # Compare linear vs quadratic
    linear_model, linear_results = fit(X_quad, y_quad, model="Linear")
    quad_model, quad_results = fit(X_quad, y_quad, model="Quadratic")
    
    print(f"Linear Model R²: {linear_results['r2_score']:.3f}")
    print(f"Quadratic Model R²: {quad_results['r2_score']:.3f}")
    print(f"Improvement: {quad_results['r2_score'] - linear_results['r2_score']:.3f}\n")
    
    # Example 3: Model comparison
    print("3. Model Comparison Example")
    print("-" * 28)
    
    X_complex = np.random.randn(n_samples, 1)
    y_complex = 0.5 * X_complex[:, 0]**3 + X_complex[:, 0]**2 + X_complex[:, 0] + np.random.randn(n_samples) * 0.3
    
    models = ["Linear", "Quadratic", "Cubic"]
    results_comparison = {}
    
    for model_type in models:
        model, result = fit(X_complex, y_complex, model=model_type)
        results_comparison[model_type] = result['r2_score']
        print(f"{model_type:>10}: R² = {result['r2_score']:.3f}")
    
    best_model = max(results_comparison, key=results_comparison.get)
    print(f"Best model: {best_model}\n")
    
    # Example 4: Feature engineering with polynomial models
    print("4. Feature Engineering Example")
    print("-" * 31)
    
    cubic = Cubic()
    cubic.fit(X_complex, y_complex)
    
    try:
        feature_names = cubic.get_feature_names(['x'])
        print(f"Polynomial features: {feature_names[:5]}...")  # Show first 5
        print(f"Total features: {len(feature_names)}")
    except Exception as e:
        print(f"Feature names not available: {e}")
    
    print(f"Model parameters: {len(cubic.get_params())} parameters")
    
    # Example 5: Predictions
    print("\n5. Making Predictions")
    print("-" * 20)
    
    # New data for prediction
    X_new = np.array([[1.0, 2.0], [0.5, -1.0], [-1.0, 0.5]])
    
    linear_pred = linear.predict(X_new)
    print("Predictions for new data:")
    for i, pred in enumerate(linear_pred):
        print(f"Sample {i+1}: {pred:.3f}")
    
    print("\n=== Examples completed successfully! ===")


if __name__ == "__main__":
    main()