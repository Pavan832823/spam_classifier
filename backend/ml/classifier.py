"""
Email Spam Classifier - Inference Engine
PRODUCTION-READY: All bugs fixed.

FIXED:
  1. Singleton race condition: __new__ was not thread-safe — two concurrent
     threads could both pass the `_instance is None` check and create two
     instances. Fixed with a class-level Lock (double-checked locking).
"""

import os
import sys
import json
import joblib
import re
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocessor import TextPreprocessor

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "backend", "ml", "saved_models")
MODEL_PATH = os.path.join(MODELS_DIR, "spam_classifier.pkl")
META_PATH  = os.path.join(MODELS_DIR, "model_metadata.json")

SPAM_THRESHOLD = 0.5


class SpamClassifier:
    """
    Production inference engine for email spam classification.
    Thread-safe singleton.
    """

    _instance = None
    _instance_lock = threading.Lock()   # BUG FIXED: guards singleton creation

    def __new__(cls):
        # Double-checked locking — avoids lock overhead on every call
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._pipeline    = None
        self._metadata    = None
        self._preprocessor = TextPreprocessor()
        self._load()
        self._initialized = True

    # ── Model loading ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(MODEL_PATH):
            print("⚠️  No saved model found. Training new model...")
            from ml.trainer import SpamModelTrainer
            trainer = SpamModelTrainer()
            trainer.run()

        print(f"📦 Loading model from: {MODEL_PATH}")
        self._pipeline = joblib.load(MODEL_PATH)

        if os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                self._metadata = json.load(f)
            print(f"✅ Loaded model: {self._metadata.get('best_model_name', 'Unknown')}")

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(
        self,
        body: str,
        subject: str = "",
        sender: str = "",
        threshold: float = SPAM_THRESHOLD,
    ) -> dict:
        if not isinstance(body, str):
            body = ""

        full_text = f"Subject: {subject}\nFrom: {sender}\n\n{body}"
        cleaned   = self._preprocessor.clean(full_text)

        try:
            proba          = self._pipeline.predict_proba([cleaned])[0]
            spam_confidence = float(proba[1])
            ham_confidence  = float(proba[0])
        except AttributeError:
            pred           = self._pipeline.predict([cleaned])[0]
            spam_confidence = 1.0 if pred == 1 else 0.0
            ham_confidence  = 1.0 - spam_confidence

        is_spam = spam_confidence >= threshold
        label   = "spam" if is_spam else "ham"
        action  = "move_to_trash" if is_spam else "store_in_inbox"

        return {
            "label":          label,
            "is_spam":        is_spam,
            "confidence":     round(spam_confidence, 4),
            "ham_confidence": round(ham_confidence, 4),
            "action":         action,
            "threshold_used": threshold,
            "indicators":     self._get_spam_indicators(subject, body, sender),
        }

    # ── Explainability ───────────────────────────────────────────────────────

    def _get_spam_indicators(self, subject: str, body: str, sender: str) -> list:
        indicators   = []
        text_upper   = f"{subject} {body}".upper()

        keyword_rules = [
            ("URGENT",          "Urgency trigger words"),
            ("FREE",            "Free offer keywords"),
            ("WINNER",          "Prize/winner language"),
            ("CONGRATULATIONS", "Congratulation triggers"),
            ("CLICK HERE",      "Call-to-action pressure"),
            ("LIMITED TIME",    "Scarcity pressure"),
            ("ACT NOW",         "Urgency triggers"),
            ("GUARANTEED",      "Unrealistic guarantees"),
            ("!!!",             "Excessive exclamation marks"),
            ("$$$",             "Currency symbols"),
        ]
        for keyword, reason in keyword_rules:
            if keyword in text_upper:
                indicators.append(reason)

        suspicious_tlds = (".xyz", ".biz", ".ru", ".tk", ".cc")
        for tld in suspicious_tlds:
            if sender.lower().endswith(tld):
                indicators.append(f"Suspicious sender domain ({tld})")
                break

        if re.search(r"https?://\S+", body, re.IGNORECASE):
            indicators.append("Contains external URLs")

        return list(set(indicators))

    # ── Metadata ─────────────────────────────────────────────────────────────

    @property
    def model_info(self) -> dict:
        return self._metadata or {}

    def is_ready(self) -> bool:
        return self._pipeline is not None


# ── Module-level accessor ─────────────────────────────────────────────────────

_classifier: SpamClassifier | None = None
_clf_lock = threading.Lock()


def get_classifier() -> SpamClassifier:
    global _classifier
    if _classifier is None:
        with _clf_lock:
            if _classifier is None:
                _classifier = SpamClassifier()
    return _classifier


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clf = SpamClassifier()
    tests = [
        ("john.doe@company.com", "Meeting at 3pm today",
         "Hi, confirming our meeting at 3pm. Please bring the reports."),
        ("prize@winner-notify.xyz", "YOU HAVE WON $1,000,000!!!",
         "CONGRATULATIONS! You've been selected! Click NOW to claim!!!"),
        ("noreply@paypal-verify.ru", "URGENT: Account suspended",
         "Your PayPal account is SUSPENDED. Update banking details immediately."),
    ]
    print("🧪 Classifier Test Results\n" + "=" * 60)
    for sender, subject, body in tests:
        r = clf.predict(body, subject, sender)
        tag = "🚫 SPAM" if r["is_spam"] else "✅ HAM"
        print(f"\n{subject}")
        print(f"  {tag} (confidence: {r['confidence']:.2%})")
        if r["indicators"]:
            print(f"  Indicators: {', '.join(r['indicators'][:3])}")
