"""
Test Suite for Email Spam Classifier
Tests: preprocessor, dataset generator, model training, classifier, database, API logic

FIXED:
  1. test_threshold_effect: original asserted confidence is EQUAL at both thresholds,
     then commented "classification may differ" — the assertion was a no-op / tautology.
     Fixed to assert that low threshold classifies as spam and high as ham for
     borderline input, and that is_spam correctly reflects the threshold.
"""

import sys
import os
import unittest
import json
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Preprocessor
# ══════════════════════════════════════════════════════════════════════════════
class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        from ml.preprocessor import TextPreprocessor
        self.p = TextPreprocessor()

    def test_lowercase(self):
        result = self.p.clean("HELLO WORLD")
        self.assertEqual(result, result.lower())

    def test_url_replacement(self):
        result = self.p.clean("Visit http://spam-site.xyz for FREE money!")
        self.assertIn("token_url", result.lower())
        self.assertNotIn("http://", result)

    def test_email_replacement(self):
        result = self.p.clean("Contact us at spam@scam.com")
        self.assertIn("token_email", result.lower())

    def test_html_strip(self):
        result = self.p.clean("<html><b>URGENT</b></html>")
        self.assertNotIn("<html>", result)
        self.assertNotIn("<b>", result)

    def test_currency_replacement(self):
        result = self.p.clean("Win $1000 today!")
        self.assertIn("token_money", result.lower())

    def test_stopwords_removal(self):
        result = self.p.clean("the cat sat on the mat")
        self.assertNotIn(" the ", f" {result} ")

    def test_empty_string(self):
        self.assertEqual(self.p.clean(""), "")

    def test_none_handling(self):
        self.assertEqual(self.p.clean(None), "")

    def test_spam_features_preserved(self):
        result = self.p.clean("CONGRATULATIONS FREE WINNER ACT NOW CLICK!!!")
        self.assertGreater(len(result), 0)

    def test_underscore_preserved_in_tokens(self):
        # TOKEN_URL underscores must survive punctuation removal
        result = self.p.clean("Go to http://x.com now")
        self.assertIn("token_url", result.lower())


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Dataset Generator
# ══════════════════════════════════════════════════════════════════════════════
class TestDatasetGenerator(unittest.TestCase):

    def _gen(self, n: int):
        sys.path.insert(0, os.path.join(BASE_DIR, "data"))
        from generate_dataset import generate_dataset
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            tmppath = f.name
        df = generate_dataset(n, tmppath)
        os.unlink(tmppath)
        return df

    def test_generates_correct_count(self):
        self.assertEqual(len(self._gen(100)), 100)

    def test_has_required_columns(self):
        df = self._gen(50)
        self.assertIn("text", df.columns)
        self.assertIn("label", df.columns)

    def test_label_values(self):
        df = self._gen(100)
        self.assertTrue(set(df["label"].unique()).issubset({0, 1}))

    def test_both_classes_present(self):
        df = self._gen(100)
        self.assertIn(0, df["label"].values)
        self.assertIn(1, df["label"].values)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Model Training
# ══════════════════════════════════════════════════════════════════════════════
class TestModelTraining(unittest.TestCase):

    def test_training_pipeline(self):
        sys.path.insert(0, os.path.join(BASE_DIR, "data"))
        from generate_dataset import generate_dataset
        from ml.trainer import SpamModelTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "test_data.csv")
            generate_dataset(200, data_path)
            trainer = SpamModelTrainer()
            df = trainer.load_data(data_path)
            df = trainer.preprocess(df)
            trainer.model_configs = dict(list(trainer.model_configs.items())[:2])
            trainer.train_and_evaluate(df)

            self.assertIsNotNone(trainer.best_pipeline)
            self.assertIsNotNone(trainer.best_model_name)
            self.assertIn(trainer.best_model_name, trainer.results)

    def test_metrics_in_range(self):
        sys.path.insert(0, os.path.join(BASE_DIR, "data"))
        from generate_dataset import generate_dataset
        from ml.trainer import SpamModelTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            generate_dataset(300, data_path)
            trainer = SpamModelTrainer()
            df = trainer.load_data(data_path)
            df = trainer.preprocess(df)
            trainer.model_configs = dict(list(trainer.model_configs.items())[:1])
            trainer.train_and_evaluate(df)

            for result in trainer.results.values():
                m = result["metrics"]
                self.assertGreaterEqual(m["f1_score"], 0.0)
                self.assertLessEqual(m["f1_score"],    1.0)
                self.assertGreaterEqual(m["accuracy"],  0.0)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Classifier Inference
# ══════════════════════════════════════════════════════════════════════════════
class TestClassifier(unittest.TestCase):

    def _get_trained_classifier(self):
        sys.path.insert(0, os.path.join(BASE_DIR, "data"))
        from generate_dataset import generate_dataset
        from ml.trainer import SpamModelTrainer
        from ml.classifier import SpamClassifier

        models_dir = os.path.join(BASE_DIR, "backend", "ml", "saved_models")
        os.makedirs(models_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            generate_dataset(400, data_path)
            trainer = SpamModelTrainer()
            df = trainer.load_data(data_path)
            df = trainer.preprocess(df)
            trainer.train_and_evaluate(df)
            trainer.save_model()

        SpamClassifier._instance = None
        return SpamClassifier()

    def test_result_structure(self):
        clf    = self._get_trained_classifier()
        result = clf.predict("Hello, let's meet at 3pm today.")
        for key in ("label", "is_spam", "confidence", "action", "indicators"):
            self.assertIn(key, result)

    def test_confidence_in_range(self):
        clf    = self._get_trained_classifier()
        result = clf.predict("You won $1000000 CLICK NOW FREE!!!")
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"],    1.0)

    def test_label_values(self):
        clf = self._get_trained_classifier()
        for body in ("Normal meeting email text", "WIN FREE MONEY NOW CLICK!!!"):
            result = clf.predict(body)
            self.assertIn(result["label"], ("spam", "ham"))

    def test_action_consistency(self):
        clf    = self._get_trained_classifier()
        result = clf.predict("Test email body")
        if result["is_spam"]:
            self.assertEqual(result["action"], "move_to_trash")
        else:
            self.assertEqual(result["action"], "store_in_inbox")

    def test_obvious_spam_detected(self):
        clf  = self._get_trained_classifier()
        spam = "CONGRATULATIONS!!! You HAVE WON $1,000,000 USD LOTTERY!!! Click http://scam.xyz NOW FREE WINNER ACT IMMEDIATELY!!!"
        result = clf.predict(spam, subject="YOU WON!!!", sender="prize@scam.xyz")
        self.assertGreater(result["confidence"], 0.3)

    def test_threshold_effect(self):
        """
        BUG FIXED: Original test asserted confidence is EQUAL for both thresholds,
        which is correct (confidence doesn't change with threshold), but then the
        comment said "classification may differ" and never tested it.
        Now we test the meaningful invariant: is_spam reflects the threshold correctly.
        """
        clf  = self._get_trained_classifier()
        body = "Moderate email: free offer but also a normal update"

        result = clf.predict(body, threshold=0.01)  # almost always spam
        self.assertTrue(result["is_spam"], "threshold=0.01 should classify almost anything as spam")

        result_low  = clf.predict(body, threshold=0.01)
        result_high = clf.predict(body, threshold=0.99)

        # Confidence score is invariant to threshold
        self.assertAlmostEqual(result_low["confidence"], result_high["confidence"])

        # But classification should differ for a borderline case
        self.assertTrue(result_low["is_spam"])
        self.assertFalse(result_high["is_spam"])


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: Database
# ══════════════════════════════════════════════════════════════════════════════
class TestDatabase(unittest.TestCase):

    def setUp(self):
        from db import database
        self.tmpdir    = tempfile.mkdtemp()
        self._orig_path = database.DB_PATH
        database.DB_PATH = os.path.join(self.tmpdir, "test.db")
        database.init_db()
        self.db = database

    def tearDown(self):
        self.db.DB_PATH = self._orig_path

    def test_store_in_inbox(self):
        row_id = self.db.store_in_inbox("test@test.com", "a@x.com", "Hello", "Body", 0.9, [])
        self.assertIsInstance(row_id, int)
        self.assertGreater(row_id, 0)

    def test_move_to_trash(self):
        row_id = self.db.move_to_trash("test@test.com", "spam@spam.com", "SPAM", "Spam body", 0.95, ["urgency"])
        self.assertIsInstance(row_id, int)

    def test_get_inbox(self):
        self.db.store_in_inbox("a@b.com", "s@s.com", "S", "B", 0.8, [])
        self.assertGreater(len(self.db.get_inbox("a@b.com")), 0)

    def test_get_trash(self):
        self.db.move_to_trash("a@b.com", "s@s.com", "Spam", "Spam body", 0.99, [])
        self.assertGreater(len(self.db.get_trash("a@b.com")), 0)

    def test_restore_to_inbox(self):
        spam_id = self.db.move_to_trash("a@b.com", "s@s.com", "Spam", "Body", 0.99, [])
        self.assertTrue(self.db.restore_to_inbox(spam_id, "a@b.com"))
        self.assertGreater(len(self.db.get_inbox("a@b.com")), 0)

    def test_delete_from_trash(self):
        spam_id = self.db.move_to_trash("a@b.com", "s@s.com", "Spam", "Body", 0.99, [])
        self.assertTrue(self.db.delete_from_trash(spam_id, "a@b.com"))
        self.assertEqual(len(self.db.get_trash("a@b.com")), 0)

    def test_stats(self):
        self.db.store_in_inbox("a@b.com", "h@h.com", "Ham",  "Body", 0.8,  [])
        self.db.move_to_trash ("a@b.com", "s@s.com", "Spam", "Body", 0.99, [])
        stats = self.db.get_stats("a@b.com")
        self.assertIn("inbox_count",  stats)
        self.assertIn("trash_count",  stats)
        self.assertGreaterEqual(stats["inbox_count"], 1)
        self.assertGreaterEqual(stats["trash_count"], 1)

    def test_empty_trash(self):
        self.db.move_to_trash("a@b.com", "s@s.com", "Spam", "Body", 0.99, [])
        self.db.empty_trash("a@b.com")
        self.assertEqual(len(self.db.get_trash("a@b.com")), 0)

    def test_indicators_serialization(self):
        indicators = ["URL detected", "Urgency keywords"]
        row_id = self.db.move_to_trash("a@b.com", "s@s.com", "S", "B", 0.9, indicators)
        email  = self.db.get_email_by_id(row_id, "trash", "a@b.com")
        self.assertEqual(email["indicators"], indicators)

    def test_uid_round_trip(self):
        self.db.update_last_uid("u@u.com", 42)
        self.assertEqual(self.db.get_last_uid("u@u.com"), 42)

    def test_log_classification(self):
        # Should not raise
        self.db.log_classification("u@u.com", "s@s.com", "Test", "spam", 0.9, "move_to_trash")


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════
def run_tests() -> bool:
    print("\n🧪 Running Email Spam Classifier Test Suite\n" + "=" * 60)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # Fast tests first
    for cls in (TestPreprocessor, TestDatasetGenerator, TestDatabase):
        suite.addTests(loader.loadTestsFromTestCase(cls))

    # Slower ML tests last
    suite.addTests(loader.loadTestsFromTestCase(TestModelTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestClassifier))

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
    return result.wasSuccessful()


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
