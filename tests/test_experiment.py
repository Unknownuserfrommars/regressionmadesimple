"""Tests for regressionmadesimple.Experiment (v4.1.0)."""

import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import regressionmadesimple as rms


class TestExperimentInit(unittest.TestCase):
    """Experiment creation and column typing."""

    def setUp(self):
        np.random.seed(42)
        n = 50
        self.df = pd.DataFrame(
            {
                "x1": np.linspace(0, 10, n),
                "x2": np.random.randn(n),
                "cat": np.random.choice(["a", "b", "c"], n),
                "flag": np.random.choice([0, 1], n),  # 0/1 → should be bool
                "is_good": np.random.choice([True, False], n),
                "y": 2 * np.linspace(0, 10, n) + np.random.randn(n) * 0.5,
            }
        )
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_init_basic(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        self.assertIsNotNone(exp)
        self.assertEqual(exp.target, "y")
        self.assertIn("train_0", exp.train_splits)

    def test_column_typing(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        self.assertIn("x1", exp._num_cols)
        self.assertIn("x2", exp._num_cols)
        self.assertIn("flag", exp._bool_cols, "0/1 column should be bool")
        self.assertIn("is_good", exp._bool_cols, "bool column should be bool")
        self.assertIn("cat", exp._cat_cols)
        self.assertTrue(
            any(c.startswith("cat_") for c in exp._dummy_cols),
            "should have one-hot dummy columns",
        )

    def test_init_missing_target(self):
        with self.assertRaises(ValueError):
            rms.Experiment(self.df, "nonexistent", self.tmpdir)


class TestExperimentTrainTestSplit(unittest.TestCase):
    """Multi-split management."""

    def setUp(self):
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {"x": np.linspace(0, 10, n), "y": np.linspace(0, 10, n) + np.random.randn(n) * 0.5}
        )
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_split_key(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        self.assertEqual(list(exp.train_splits.keys()), ["train_0"])

    def test_multiple_splits(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        key2 = exp.train_test_split(test_size=0.2, random_state=99)
        self.assertEqual(key2, "train_1")
        self.assertIn("train_0", exp.train_splits)
        self.assertIn("train_1", exp.train_splits)

    def test_validation_split(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        exp.train_test_split(test_size=0.1, validation_size=0.1, random_state=42)
        key = list(exp.train_splits.keys())[-1]
        val = exp.train_splits[key]
        self.assertIsNotNone(val.get("X_validation"))
        self.assertIsNotNone(val.get("y_validation"))
        meta = val["meta"]
        self.assertAlmostEqual(meta["validation_size"], 0.1)

    def test_split_invalid_sizes(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        with self.assertRaises(AssertionError):
            exp.train_test_split(test_size=0.6)
        with self.assertRaises(AssertionError):
            exp.train_test_split(test_size=0.3, validation_size=0.3)


class TestExperimentFitModels(unittest.TestCase):
    """Model fitting with fit_models."""

    def setUp(self):
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                "x": np.linspace(0, 10, n),
                "cat": np.random.choice(["a", "b"], n),
                "y": 2 * np.linspace(0, 10, n) + np.random.randn(n) * 0.5,
            }
        )
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_fit_single_model(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        exp.fit_models({"lr": [LinearRegression()]})
        self.assertIn("lr_0", exp.model_results)
        res = exp.model_results["lr_0"]
        self.assertIsNotNone(res["model"])
        self.assertIn("r2_score", res["metrics"]["test"])

    def test_fit_multiple_models(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        exp.fit_models(
            {
                "lr": [LinearRegression()],
                "knn": [KNeighborsRegressor(n_neighbors=3)],
            }
        )
        self.assertIn("lr_0", exp.model_results)
        self.assertIn("knn_0", exp.model_results)

    def test_fit_multiple_of_same_stem(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        exp.fit_models(
            {
                "rf": [
                    RandomForestRegressor(n_estimators=10, random_state=42),
                    RandomForestRegressor(n_estimators=50, random_state=42),
                ]
            }
        )
        self.assertIn("rf_0", exp.model_results)
        self.assertIn("rf_1", exp.model_results)

    def test_fit_with_scaler(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        exp.fit_models({"lr": [LinearRegression()]}, scaler="standard")
        res = exp.model_results["lr_0"]
        self.assertEqual(res["scaler"], "standard")

    def test_fit_invalid_scaler(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        with self.assertRaises(ValueError):
            exp.fit_models({"lr": [LinearRegression()]}, scaler="invalid")

    def test_fit_auto_increment_keys(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        exp.fit_models({"lr": [LinearRegression()]})
        exp.fit_models({"lr": [LinearRegression()]})  # same config → dedup
        # Second call should be deduplicated
        self.assertIn("lr_0", exp.model_results)
        # But if we change config, a new key should be created
        exp.fit_models({"rf": [RandomForestRegressor(random_state=42)]})
        self.assertIn("rf_0", exp.model_results)

    def test_fit_with_categorical(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        exp.fit_models({"lr": [LinearRegression()]})
        res = exp.model_results["lr_0"]
        # Should have dummy columns from 'cat'
        dummies_in_cols = [c for c in res["feature_cols"] if c.startswith("cat_")]
        self.assertTrue(len(dummies_in_cols) > 0)

    def test_fit_column_selection_explicit(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        exp.fit_models({"lr": [LinearRegression()]}, columns=["x"])
        res = exp.model_results["lr_0"]
        self.assertEqual(res["feature_cols"], ["x"])

    def test_fit_column_selection_cat(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        exp.fit_models({"lr": [LinearRegression()]}, columns=["cat"])
        res = exp.model_results["lr_0"]
        dummies = [c for c in res["feature_cols"] if c.startswith("cat_")]
        self.assertTrue(len(dummies) > 0)

    def test_fit_invalid_column(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        with self.assertRaises(ValueError):
            exp.fit_models({"lr": [LinearRegression()]}, columns=["ghost_col"])

    def test_fit_invalid_split(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        with self.assertRaises(ValueError):
            exp.fit_models(
                {"lr": [LinearRegression()]}, train_split="nonexistent"
            )

    def test_regression_metrics_populated(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        exp.fit_models({"lr": [LinearRegression()]})
        res = exp.model_results["lr_0"]
        test_metrics = res["metrics"]["test"]
        train_metrics = res["metrics"]["train"]
        for key in ("r2_score", "mean_squared_error", "rmse", "mean_absolute_error"):
            self.assertIn(key, test_metrics, f"Missing metric: {key}")
            self.assertIn(key, train_metrics, f"Missing metric: {key}")


class TestExperimentPersistence(unittest.TestCase):
    """Save / load via dill."""

    def setUp(self):
        np.random.seed(42)
        n = 50
        self.df = pd.DataFrame(
            {"x": np.linspace(0, 10, n), "y": np.linspace(0, 10, n) + np.random.randn(n) * 0.5}
        )
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        exp.fit_models({"lr": [LinearRegression()]})
        path = exp.save()
        self.assertTrue(path.exists())

        loaded = rms.Experiment.load(path)
        self.assertEqual(loaded.target, "y")
        self.assertIn("lr_0", loaded.model_results)
        self.assertIn("train_0", loaded.train_splits)

    def test_loaded_model_still_predicts(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        exp.fit_models({"lr": [LinearRegression()]})
        path = exp.save()
        loaded = rms.Experiment.load(path)
        res = loaded.model_results["lr_0"]
        model = res["model"]
        X_test = loaded.train_splits["train_0"]["X_test"][res["feature_cols"]]
        preds = model.predict(X_test)
        self.assertEqual(len(preds), len(X_test))

    def test_default_save_path(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        path = exp.save()
        self.assertEqual(path, self.tmpdir / "experiment.dill")

    def test_custom_save_path(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        custom = self.tmpdir / "my_exp.dill"
        path = exp.save(custom)
        self.assertEqual(path, custom)

    def test_load_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            rms.Experiment.load(self.tmpdir / "ghost.dill")


class TestExperimentAddColumns(unittest.TestCase):
    """Iterative feature engineering with add_columns."""

    def setUp(self):
        np.random.seed(42)
        n = 50
        self.df = pd.DataFrame(
            {"x": np.linspace(0, 10, n), "y": np.linspace(0, 10, n) + np.random.randn(n) * 0.5}
        )
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_column_propagates_to_df_and_splits(self):
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        x_squared = self.df[["x"]].assign(x2=self.df["x"] ** 2)[["x2"]]
        exp.add_columns(x_squared)
        self.assertIn("x2", exp.df.columns)
        self.assertIn("x2", exp.train_splits["train_0"]["X_train"].columns)

    def test_add_column_retrains_models(self):
        """After adding a column, re-running fit_models should include it by default."""
        exp = rms.Experiment(self.df, "y", self.tmpdir, random_state=42)
        x_squared = self.df[["x"]].assign(x2=self.df["x"] ** 2)[["x2"]]
        exp.add_columns(x_squared)
        exp.fit_models({"lr": [LinearRegression()]})
        res = exp.model_results["lr_0"]
        self.assertIn("x2", res["feature_cols"])


class TestExperimentEdgeCases(unittest.TestCase):
    """Edge cases and robustness."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_out_path_created(self):
        path = self.tmpdir / "nested" / "dir"
        df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        exp = rms.Experiment(df, "y", path)
        self.assertTrue(Path(path).exists())

    def test_with_bool_target(self):
        df = pd.DataFrame(
            {"x": [1, 2, 3, 4, 5], "y": [0, 1, 0, 1, 0]}
        )
        exp = rms.Experiment(df, "y", self.tmpdir, random_state=42)
        exp.fit_models({"lr": [LogisticRegression()]})
        self.assertIn("lr_0", exp.model_results)

    def test_empty_cat_cols(self):
        df = pd.DataFrame({"x": range(10), "y": range(10)})
        exp = rms.Experiment(df, "y", self.tmpdir, random_state=42)
        self.assertEqual(exp._cat_cols, [])
        self.assertEqual(exp._dummy_cols, [])

    def test_bool_cols_dedup(self):
        """Ensure bool columns aren't duplicated in _bool_cols."""
        df = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "flag": [0, 1, 0],
                "is_good": [True, False, True],
                "y": [2, 4, 6],
            }
        )
        exp = rms.Experiment(df, "y", self.tmpdir, random_state=42)
        self.assertEqual(exp._bool_cols, ["is_good", "flag"])

    def test_dedup_identical_configs(self):
        df = pd.DataFrame({"x": range(20), "y": [2*v + 1 for v in range(20)]})
        exp = rms.Experiment(df, "y", self.tmpdir, random_state=42)
        exp.fit_models({"lr": [LinearRegression()]})
        exp.fit_models({"lr": [LinearRegression()]})  # identical → dedup
        self.assertEqual(len(exp.model_results), 1)  # should be dedup'd


class TestExperimentExamplesFromDocstring(unittest.TestCase):
    """Verify examples from the docstring actually run."""

    def test_quickstart_example(self):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "cat": ["a", "b", "a", "b", "a"],
                "y": [2.1, 4.2, 6.1, 8.3, 10.2],
            }
        )
        tmpdir = Path(tempfile.mkdtemp())
        try:
            exp = rms.Experiment(df, target="y", out_path=tmpdir)
            exp.fit_models({"lr": [LinearRegression()]})
            path = exp.save()
            loaded = rms.Experiment.load(path)
            self.assertIsNotNone(loaded.model_results["lr_0"]["model"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestLOWER_IS_BETTER(unittest.TestCase):
    """Test the _LOWER_IS_BETTER frozenset."""

    def test_common_metrics(self):
        self.assertIn("mean_squared_error", rms._LOWER_IS_BETTER)
        self.assertIn("rmse", rms._LOWER_IS_BETTER)
        self.assertIn("mean_absolute_error", rms._LOWER_IS_BETTER)
        self.assertIn("log_loss", rms._LOWER_IS_BETTER)
        self.assertNotIn("r2_score", rms._LOWER_IS_BETTER)
        self.assertNotIn("accuracy", rms._LOWER_IS_BETTER)


if __name__ == "__main__":
    unittest.main()
