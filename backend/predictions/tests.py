from unittest.mock import Mock

import numpy as np
from django.test import SimpleTestCase

from .model_service import SkinCancerModelService
from .preprocessing import build_metadata_row


class ScreeningInputTests(SimpleTestCase):
    def test_form_values_match_training_categories(self):
        row = build_metadata_row(
            {"age": 52.0, "sex": "female", "anatom_site": "face", "smoke": "false", "drink": "true"},
            ["age", "gender", "region", "smoke", "drink"],
        )
        self.assertEqual(row, {"age": 52.0, "gender": "FEMALE", "region": "FACE", "smoke": False, "drink": True})

    def test_unknown_values_use_imputer_missing_value(self):
        row = build_metadata_row({"sex": "", "smoke": "unknown"}, ["age", "gender", "smoke"])
        self.assertTrue(all(np.isnan(value) for value in row.values()))

    def test_inference_uses_one_image_and_one_metadata_row(self):
        service = object.__new__(SkinCancerModelService)
        service.metadata_schema = {"feature_columns": ["age"]}
        service.metadata_preprocessor = Mock()
        service.metadata_preprocessor.transform.return_value = np.array([[52.0]])
        service.model = Mock()
        service.model.predict.return_value = np.array([[0.25, 0.75]])
        images = np.zeros((1, 300, 300, 3), dtype=np.float32)
        result = service._real_predict(images, {"age": 52.0})
        inputs = service.model.predict.call_args.args[0]
        self.assertEqual(inputs["image_input"].shape[0], 1)
        self.assertEqual(inputs["metadata_input"].shape[0], 1)
        np.testing.assert_allclose(result, [0.25, 0.75])

    def test_only_legacy_schema_gets_biopsy_warning(self):
        service = object.__new__(SkinCancerModelService)
        service.metadata_schema = {"feature_columns": ["age", "biopsed"]}
        self.assertTrue(service.model_warnings())
        service.metadata_schema = {"feature_columns": ["age", "gender", "region", "smoke", "drink"]}
        self.assertEqual(service.model_warnings(), [])

    def test_observed_melanoma_failure_is_disclosed(self):
        service = object.__new__(SkinCancerModelService)
        service.metadata_schema = None
        service.mode = "real"
        service.evaluation_metrics = {"classification_report": {"MEL": {"support": 5, "recall": 0.0}}}
        self.assertIn("missed all 5 melanoma", service.model_warnings()[0])

    def test_demo_does_not_claim_the_bundled_models_evaluation(self):
        service = object.__new__(SkinCancerModelService)
        service.metadata_schema = None
        service.mode = "demo"
        service.evaluation_metrics = {"classification_report": {"MEL": {"support": 5, "recall": 0.0}}}
        self.assertEqual(service.model_warnings(), [])
