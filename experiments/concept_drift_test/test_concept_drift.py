import os
import unittest
from unittest.mock import patch

import concept_drift


class ConfigAndInputTests(unittest.TestCase):
    def test_missing_db_config_fails_before_connect(self):
        with patch.object(concept_drift, "DB_HOST", None), \
             patch.object(concept_drift, "DB_NAME", None), \
             patch.object(concept_drift, "DB_USER", None), \
             patch.object(concept_drift, "DB_PASSWORD", None):
            with self.assertRaisesRegex(RuntimeError, "Missing required database"):
                concept_drift.db_connect()

    def test_normalize_string_true(self):
        got = concept_drift.normalize_model_entry({
            "model_id": 9, "model_name": "ppo-gnn", "degraded": "true"
        })
        self.assertEqual(got, {
            "model_id": 9, "model_name": "ppo-gnn", "degraded": True
        })

    def test_normalize_false(self):
        got = concept_drift.normalize_model_entry({
            "model_id": 1, "degraded": False
        })
        self.assertEqual(got["degraded"], False)
        self.assertEqual(got["model_name"], "model-1")

    def test_non_degraded_does_not_touch_database_or_gemini(self):
        with patch.object(concept_drift, "db_connect", side_effect=AssertionError("DB called")), \
             patch.object(concept_drift, "build_graph", side_effect=AssertionError("Gemini called")):
            got = concept_drift.run_autonomous_investigation(1, "test", False)
        self.assertEqual(got["status"], "not_investigated")
        self.assertFalse(got["concept_drift_detected"])
        self.assertEqual(got["severity_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
