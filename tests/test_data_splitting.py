#!/usr/bin/env python3
"""
Tests for data splitting functionality in RegressionMadeSimple package.
"""

import pytest
import numpy as np
from regressionmadesimple import fit, Linear, Quadratic, Cubic
from regressionmadesimple.utils import validate_split_ratio, split_data
import tempfile
import os


class TestDataSplitting:
    """Test data splitting functionality."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.randn(100, 2)
        self.y = 2 * self.X[:, 0] + 3 * self.X[:, 1] + np.random.randn(100) * 0.1
    
    def test_validate_split_ratio(self):
        """Test split ratio validation."""
        # Valid ratios
        assert validate_split_ratio([8, 2]) == [0.8, 0.2]
        assert validate_split_ratio([7, 3]) == [0.7, 0.3]
        assert validate_split_ratio([0.8, 0.2]) == [0.8, 0.2]
        
        # Invalid ratios
        with pytest.raises(ValueError):
            validate_split_ratio([8])  # Wrong length
        with pytest.raises(ValueError):
            validate_split_ratio([8, 2, 1])  # Wrong length
        with pytest.raises(ValueError):
            validate_split_ratio([0, 2])  # Zero value
        with pytest.raises(ValueError):
            validate_split_ratio([-1, 2])  # Negative value
    
    def test_split_data_utility(self):
        """Test the split_data utility function."""
        X_train, X_test, y_train, y_test = split_data(self.X, self.y, [8, 2], random_state=42)
        
        # Check shapes
        assert X_train.shape[0] + X_test.shape[0] == self.X.shape[0]
        assert y_train.shape[0] + y_test.shape[0] == self.y.shape[0]
        assert X_train.shape[1] == self.X.shape[1]
        assert X_test.shape[1] == self.X.shape[1]
        
        # Check approximate split ratio
        total_samples = len(self.y)
        expected_train_size = int(0.8 * total_samples)
        assert abs(len(y_train) - expected_train_size) <= 1  # Allow for rounding
    
    def test_function_api_with_splitting(self):
        """Test function API with data splitting."""
        model, results = fit(self.X, self.y, model="Linear", split_ratio=[8, 2], random_state=42)
        
        # Check results structure
        assert 'train_r2_score' in results
        assert 'test_r2_score' in results
        assert 'n_samples_train' in results
        assert 'n_samples_test' in results
        assert 'split_ratio' in results
        assert 'random_state' in results
        
        # Check model attributes
        assert hasattr(model, 'X_train')
        assert hasattr(model, 'y_train')
        assert hasattr(model, 'X_test')
        assert hasattr(model, 'y_test')
        assert model.X_train is not None
        assert model.X_test is not None
        
        # Check split ratio is preserved
        assert results['split_ratio'] == [8, 2]
        assert results['random_state'] == 42
    
    def test_linear_model_with_splitting(self):
        """Test Linear model with data splitting."""
        model = Linear(split_ratio=[8, 2], random_state=42)
        model.fit(self.X, self.y)
        
        assert model.is_fitted
        assert model.X_train is not None
        assert model.X_test is not None
        assert model.y_train is not None
        assert model.y_test is not None
        
        # Test scores
        train_score = model.get_train_score()
        test_score = model.get_test_score()
        assert train_score is not None
        assert test_score is not None
        assert 0 <= train_score <= 1
        assert 0 <= test_score <= 1
        
        # Test parameters include split info
        params = model.get_params()
        assert 'split_ratio' in params
        assert 'n_samples_train' in params
        assert 'n_samples_test' in params
        assert 'train_r2_score' in params
        assert 'test_r2_score' in params
    
    def test_quadratic_model_with_splitting(self):
        """Test Quadratic model with data splitting."""
        model = Quadratic(split_ratio=[7, 3], random_state=123)
        model.fit(self.X, self.y)
        
        assert model.is_fitted
        assert model.X_test is not None
        assert model.get_test_score() is not None
        
        # Check split ratio in string representation
        assert "split_ratio=[7, 3]" in str(model)
    
    def test_cubic_model_with_splitting(self):
        """Test Cubic model with data splitting."""
        model = Cubic(split_ratio=[9, 1], random_state=456)
        model.fit(self.X, self.y)
        
        assert model.is_fitted
        assert model.X_test is not None
        assert model.get_test_score() is not None
        
        # Test feature names work with split data
        feature_names = model.get_feature_names(['x1', 'x2'])
        assert len(feature_names) > 0
    
    def test_model_without_splitting(self):
        """Test that models work normally without splitting."""
        model = Linear()  # No split_ratio
        model.fit(self.X, self.y)
        
        assert model.is_fitted
        assert model.X_train is not None  # Should still have training data
        assert model.X_test is None  # Should not have test data
        assert model.get_test_score() is None
        assert model.get_train_score() is not None
    
    def test_different_split_ratios(self):
        """Test different split ratios."""
        ratios = [[9, 1], [8, 2], [7, 3], [6, 4]]
        
        for ratio in ratios:
            model, results = fit(self.X, self.y, model="Linear", split_ratio=ratio, random_state=42)
            
            expected_train_ratio = ratio[0] / sum(ratio)
            actual_train_ratio = results['n_samples_train'] / (results['n_samples_train'] + results['n_samples_test'])
            
            # Allow small difference due to rounding
            assert abs(expected_train_ratio - actual_train_ratio) < 0.05
    
    def test_decimal_split_ratios(self):
        """Test decimal split ratios."""
        model, results = fit(self.X, self.y, model="Linear", split_ratio=[0.8, 0.2], random_state=42)
        
        assert 'train_r2_score' in results
        assert 'test_r2_score' in results
        assert model.X_test is not None
    
    def test_reproducible_splits(self):
        """Test that splits are reproducible with same random_state."""
        model1, results1 = fit(self.X, self.y, model="Linear", split_ratio=[8, 2], random_state=42)
        model2, results2 = fit(self.X, self.y, model="Linear", split_ratio=[8, 2], random_state=42)
        
        # Should have same split
        np.testing.assert_array_equal(model1.X_train, model2.X_train)
        np.testing.assert_array_equal(model1.X_test, model2.X_test)
        np.testing.assert_array_equal(model1.y_train, model2.y_train)
        np.testing.assert_array_equal(model1.y_test, model2.y_test)
    
    def test_different_random_states(self):
        """Test that different random states give different splits."""
        model1, results1 = fit(self.X, self.y, model="Linear", split_ratio=[8, 2], random_state=42)
        model2, results2 = fit(self.X, self.y, model="Linear", split_ratio=[8, 2], random_state=123)
        
        # Should have different splits
        assert not np.array_equal(model1.X_train, model2.X_train)
    
    def test_model_comparison_with_splitting(self):
        """Test model comparison with data splitting."""
        models = ["Linear", "Quadratic", "Cubic"]
        results = {}
        
        for model_type in models:
            model, result = fit(self.X, self.y, model=model_type, split_ratio=[8, 2], random_state=42)
            results[model_type] = result
            
            # All should have split data
            assert 'train_r2_score' in result
            assert 'test_r2_score' in result
            assert model.X_test is not None
    
    def test_save_load_model_with_split_data(self):
        """Test saving and loading models with split data."""
        model = Linear(split_ratio=[8, 2], random_state=42)
        model.fit(self.X, self.y)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            # Save model
            from regressionmadesimple.utils import save_model, load_model
            save_model(model, tmp_path)
            
            # Load model
            loaded_model = load_model(tmp_path)
            
            # Check split data is preserved
            assert loaded_model.split_ratio == [8, 2]
            assert loaded_model.random_state == 42
            assert loaded_model.X_test is not None
            assert loaded_model.get_test_score() is not None
            
            # Test predictions match
            np.testing.assert_array_almost_equal(
                model.predict(self.X), 
                loaded_model.predict(self.X)
            )
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestEdgeCasesWithSplitting:
    """Test edge cases with data splitting."""
    
    def test_small_dataset_splitting(self):
        """Test splitting with very small datasets."""
        X_small = np.random.randn(10, 2)
        y_small = np.random.randn(10)
        
        # Should still work but might have very small test set
        model, results = fit(X_small, y_small, model="Linear", split_ratio=[8, 2], random_state=42)
        
        assert results['n_samples_train'] > 0
        assert results['n_samples_test'] > 0
        assert results['n_samples_train'] + results['n_samples_test'] == 10
    
    def test_extreme_split_ratios(self):
        """Test extreme split ratios."""
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        
        # Very small test set
        model, results = fit(X, y, model="Linear", split_ratio=[99, 1], random_state=42)
        assert results['n_samples_test'] >= 1
        
        # Very small training set
        model, results = fit(X, y, model="Linear", split_ratio=[1, 99], random_state=42)
        assert results['n_samples_train'] >= 1


if __name__ == "__main__":
    pytest.main([__file__])