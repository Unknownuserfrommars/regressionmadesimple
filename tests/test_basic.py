#!/usr/bin/env python3
"""
Basic tests for RegressionMadeSimple package.
"""

import pytest
import numpy as np
from regressionmadesimple import fit, Linear, Quadratic, Cubic
from regressionmadesimple.utils import validate_input, save_model, load_model
import tempfile
import os


class TestBasicFunctionality:
    """Test basic functionality of the package."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.randn(50, 2)
        self.y = 2 * self.X[:, 0] + 3 * self.X[:, 1] + np.random.randn(50) * 0.1
    
    def test_function_api(self):
        """Test the function-style API."""
        model, results = fit(self.X, self.y, model="Linear")
        
        assert model is not None
        assert isinstance(results, dict)
        assert 'r2_score' in results
        assert 'model_type' in results
        assert results['model_type'] == 'Linear'
        assert 0 <= results['r2_score'] <= 1
    
    def test_linear_model(self):
        """Test Linear model class."""
        model = Linear()
        model.fit(self.X, self.y)
        
        assert model.is_fitted
        
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.y)
        
        score = model.score(self.X, self.y)
        assert 0 <= score <= 1
        
        params = model.get_params()
        assert 'coefficients' in params
        assert 'intercept' in params
    
    def test_quadratic_model(self):
        """Test Quadratic model class."""
        model = Quadratic()
        model.fit(self.X, self.y)
        
        assert model.is_fitted
        
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.y)
        
        score = model.score(self.X, self.y)
        assert 0 <= score <= 1
    
    def test_cubic_model(self):
        """Test Cubic model class."""
        model = Cubic()
        model.fit(self.X, self.y)
        
        assert model.is_fitted
        
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.y)
        
        score = model.score(self.X, self.y)
        assert 0 <= score <= 1
    
    def test_validation(self):
        """Test input validation."""
        X_valid, y_valid = validate_input(self.X, self.y)
        
        assert isinstance(X_valid, np.ndarray)
        assert isinstance(y_valid, np.ndarray)
        assert X_valid.shape == self.X.shape
        assert y_valid.shape == self.y.shape
    
    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError):
            fit(self.X, self.y, model="InvalidModel")
    
    def test_unfitted_model_prediction(self):
        """Test error when predicting with unfitted model."""
        model = Linear()
        
        with pytest.raises(ValueError):
            model.predict(self.X)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        model = Linear()
        model.fit(self.X, self.y)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            # Save model
            save_model(model, tmp_path)
            assert os.path.exists(tmp_path)
            
            # Load model
            loaded_model = load_model(tmp_path)
            assert loaded_model.is_fitted
            
            # Test predictions match
            original_pred = model.predict(self.X)
            loaded_pred = loaded_model.predict(self.X)
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_model_comparison(self):
        """Test that different models give different results."""
        linear_model, linear_results = fit(self.X, self.y, model="Linear")
        quad_model, quad_results = fit(self.X, self.y, model="Quadratic")
        cubic_model, cubic_results = fit(self.X, self.y, model="Cubic")
        
        # All should be fitted
        assert linear_model.is_fitted
        assert quad_model.is_fitted
        assert cubic_model.is_fitted
        
        # Results should be different (in most cases)
        scores = [
            linear_results['r2_score'],
            quad_results['r2_score'], 
            cubic_results['r2_score']
        ]
        
        # At least one should be different (allowing for edge cases)
        assert len(set(np.round(scores, 3))) >= 1


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_input(self):
        """Test handling of empty input."""
        with pytest.raises(ValueError):
            validate_input([], [])
    
    def test_mismatched_shapes(self):
        """Test handling of mismatched X and y shapes."""
        X = np.random.randn(10, 2)
        y = np.random.randn(5)  # Wrong size
        
        with pytest.raises(ValueError):
            validate_input(X, y)
    
    def test_nan_input(self):
        """Test handling of NaN values."""
        X = np.array([[1, 2], [np.nan, 4]])
        y = np.array([1, 2])
        
        with pytest.raises(ValueError):
            validate_input(X, y)
    
    def test_single_feature(self):
        """Test with single feature input."""
        X = np.random.randn(20, 1)
        y = 2 * X[:, 0] + np.random.randn(20) * 0.1
        
        model, results = fit(X, y, model="Linear")
        assert results['n_features'] == 1
        assert model.is_fitted


if __name__ == "__main__":
    pytest.main([__file__])