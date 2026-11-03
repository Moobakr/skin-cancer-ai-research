import unittest

import pandas as pd

from .data import make_patient_level_split, metadata_feature_columns


class ScreeningDataTests(unittest.TestCase):
    def test_only_app_inputs_are_selected(self):
        frame = pd.DataFrame(columns=[
            "diagnostic", "patient_id", "biopsed", "new_diagnostic_feature",
            "age", "gender", "region", "smoke", "drink",
        ])
        self.assertEqual(metadata_feature_columns(frame, "diagnostic"),
                         ["age", "gender", "region", "smoke", "drink"])

    def test_patient_images_stay_in_one_split(self):
        frame = pd.DataFrame({
            "patient_id": [f"patient_{i // 2}" for i in range(100)],
            "diagnostic": ["NEV", "BCC"] * 50,
        })
        splits = make_patient_level_split(frame, "diagnostic", "patient_id")
        groups = [set(part.patient_id) for part in splits]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertEqual(sum(map(len, splits)), len(frame))


if __name__ == "__main__":
    unittest.main()
