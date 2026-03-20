#!/usr/bin/env python3
"""
SpamGuard — Email Spam Classifier
Master startup script.

Usage:
    python run.py          — Train model (if needed) + start server
    python run.py --train  — Force retrain model
    python run.py --test   — Run test suite
    python run.py --demo   — Classify demo emails from CLI
"""

import sys
import os
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

MODEL_PATH = os.path.join(BASE_DIR, "backend", "ml", "saved_models", "spam_classifier.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "emails.csv")


def ensure_data():
    """Generate dataset if not present."""
    if not os.path.exists(DATA_PATH):
        print("📊 Generating training dataset...")
        sys.path.insert(0, os.path.join(BASE_DIR, "data"))
        from generate_dataset import generate_dataset
        generate_dataset(2000, DATA_PATH)
    else:
        print(f"✅ Dataset found: {DATA_PATH}")


def train_model():
    """Train and save the best model."""
    print("\n🚀 Training spam classifier...")
    sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
    from ml.trainer import SpamModelTrainer
    trainer = SpamModelTrainer()
    trainer.run()


def start_server(host="0.0.0.0", port=8000):
    """Start the API server."""
    print(f"\n🌐 Starting server on http://{host}:{port}")
    print(f"   Open frontend: http://localhost:{port}/")
    print(f"   API docs:      http://localhost:{port}/api/stats")
    print("\n   Press Ctrl+C to stop\n")
    sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
    from api.server import run
    run(host=host, port=port)


def run_demo():
    """Classify a few demo emails from CLI."""
    sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
    from ml.classifier import SpamClassifier

    clf = SpamClassifier()

    demo_emails = [
        {
            "sender":  "john@company.com",
            "subject": "Project meeting tomorrow",
            "body":    "Hi team, just a reminder about our project meeting tomorrow at 10am. Please review the attached agenda.",
        },
        {
            "sender":  "prize@winner-notify.xyz",
            "subject": "YOU HAVE WON $1,000,000!!!",
            "body":    "CONGRATULATIONS! You have been selected as our lucky winner! Click http://claim-now.xyz to claim your PRIZE!!! ACT NOW before it expires!!!",
        },
        {
            "sender":  "noreply@paypal-secure.ru",
            "subject": "Your account has been SUSPENDED",
            "body":    "URGENT: Your PayPal account has been limited. Provide your banking details immediately or your account will be permanently deleted.",
        },
        {
            "sender":  "hr@startup.io",
            "subject": "Interview scheduled",
            "body":    "We'd like to invite you for a technical interview next Monday at 2pm. Please confirm your availability.",
        },
    ]

    print("\n📧 Demo Email Classification")
    print("=" * 60)
    for i, email in enumerate(demo_emails, 1):
        result = clf.predict(email["body"], email["subject"], email["sender"])
        status = "🚫 SPAM" if result["is_spam"] else "✅ HAM"
        print(f"\n[{i}] {email['subject'][:50]}")
        print(f"    From:   {email['sender']}")
        print(f"    Result: {status}")
        print(f"    Confidence: {result['confidence']:.2%}")
        print(f"    Action: {result['action']}")
        if result["indicators"]:
            print(f"    Flags:  {', '.join(result['indicators'][:2])}")


def run_tests():
    """Run the test suite."""
    print("\n🧪 Running tests...")
    sys.path.insert(0, BASE_DIR)
    from tests.test_all import run_tests as _run_tests
    return _run_tests()


def main():
    parser = argparse.ArgumentParser(description="SpamGuard Email Classifier")
    parser.add_argument("--train",  action="store_true", help="Force retrain model")
    parser.add_argument("--test",   action="store_true", help="Run test suite")
    parser.add_argument("--demo",   action="store_true", help="Run CLI demo")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Server port")
    parser.add_argument("--host",   type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    print("🛡️  SpamGuard — AI Email Spam Classifier")
    print("=" * 60)

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    # Ensure data exists
    ensure_data()

    # Train if forced or model missing
    if args.train or not os.path.exists(MODEL_PATH):
        train_model()

    if args.demo:
        run_demo()
        return

    # Default: start server
    start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
