import unittest
import numpy as np
import pandas as pd

import regressionmadesimple as rms
from regressionmadesimple.api import fit as api_fit


class TestRMSV4(unittest.TestCase):
    def setUp(self):
        self.dataset = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2.1, 4.2, 6.1, 8.3, 10.2, 12.1, 14.3, 16.2, 18.1, 20.3],
            }
        )

    def test_wrapper_rejects_string_model_in_v4(self):
        with self.assertRaises(TypeError):
            rms.LinearRegressionModel.fit(self.dataset, "x", "y", model="linear")

    def test_quadratic_model_reads_quadratic_options(self):
        model = rms.models.Quadratic(self.dataset, "x", "y", train_test_split=False)
        preds = model.predict(pd.DataFrame({"x": [11, 12]}))
        self.assertEqual(len(preds), 2)

    def test_function_api_fit_with_split(self):
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=float)
        y = np.array([2.1, 4.2, 6.1, 8.3, 10.2, 12.1, 14.3, 16.2, 18.1, 20.3], dtype=float)

        model, results = api_fit(
            X,
            y,
            model=rms.models.Linear,
            split_ratio=[8, 2],
            random_state=42,
        )

        self.assertTrue(hasattr(model, "model"))
        self.assertIn("train_r2_score", results)
        self.assertIn("test_r2_score", results)
        self.assertIn("summary", results)


if __name__ == "__main__":
    unittest.main()
