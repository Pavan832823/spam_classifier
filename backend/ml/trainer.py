"""
Model Training Pipeline for Email Spam Classifier
Trains and evaluates: Naive Bayes, Logistic Regression, SVM, Random Forest
Saves the best model based on F1 score.

FIXED:
  1. TF-IDF comment said "top 10k features" but max_features=20000 — corrected.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocessor import TextPreprocessor

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "backend", "ml", "saved_models")
DATA_PATH  = os.path.join(BASE_DIR, "data", "emails.csv")


class SpamModelTrainer:
    """
    Trains multiple spam classifiers, evaluates them, and saves the best.

    Models:
      1. Naive Bayes (MultinomialNB)      — fast probabilistic baseline
      2. Logistic Regression              — strong linear baseline
      3. Linear SVM (calibrated)          — high accuracy for text
      4. Random Forest                    — ensemble non-linear baseline

    Feature extraction: TF-IDF with unigrams + trigrams, 20 000 features.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        os.makedirs(MODELS_DIR, exist_ok=True)

        self.model_configs = {
            "naive_bayes": {
                "name": "Multinomial Naive Bayes",
                "clf":  MultinomialNB(alpha=0.1),
            },
            "logistic_regression": {
                "name": "Logistic Regression",
                "clf":  LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            },
            "svm": {
                "name": "Linear SVM (Calibrated)",
                "clf":  CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=1000, random_state=42)),
            },
            "random_forest": {
                "name": "Random Forest",
                "clf":  RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
            },
        }

        # FIXED: comment updated to match actual max_features value (20 000)
        self.tfidf_params = {
            "ngram_range":  (1, 3),
            "max_features": 20_000,   # top 20k features
            "min_df":       3,
            "max_df":       0.9,
            "sublinear_tf": True,
        }

        self.results:          dict = {}
        self.best_model_name:  str  = None
        self.best_pipeline          = None

    def load_data(self, data_path: str = DATA_PATH) -> pd.DataFrame:
        print(f"📂 Loading data from: {data_path}")
        if not os.path.exists(data_path):
            print("⚠️  Dataset not found. Generating synthetic dataset...")
            sys.path.insert(0, os.path.join(BASE_DIR, "data"))
            from generate_dataset import generate_dataset
            generate_dataset(2000, data_path)

        df = pd.read_csv(data_path)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns")
        df = df.dropna(subset=["text", "label"])
        print(f"✅ Loaded {len(df)} emails — Ham: {(df['label']==0).sum()}, Spam: {(df['label']==1).sum()}")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n🔧 Preprocessing text...")
        df = df.copy()
        df["cleaned_text"] = df["text"].apply(self.preprocessor.clean)
        print(f"✅ Preprocessed {len(df)} emails")
        return df

    def train_and_evaluate(self, df: pd.DataFrame):
        X = df["cleaned_text"].values
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}\n" + "=" * 60)

        best_f1 = -1.0

        for key, config in self.model_configs.items():
            print(f"\n🔄 Training: {config['name']}")
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(**self.tfidf_params)),
                ("clf",   config["clf"]),
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            try:
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                auc    = round(roc_auc_score(y_test, y_prob), 4)
            except AttributeError:
                auc = "N/A"

            metrics = {
                "accuracy":         round(accuracy_score(y_test, y_pred),  4),
                "precision":        round(precision_score(y_test, y_pred), 4),
                "recall":           round(recall_score(y_test, y_pred),    4),
                "f1_score":         round(f1_score(y_test, y_pred),        4),
                "roc_auc":          auc,
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }
            self.results[key] = {"name": config["name"], "metrics": metrics, "pipeline": pipeline}

            print(f"   Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1:        {metrics['f1_score']:.4f}")
            print(f"   ROC-AUC:   {metrics['roc_auc']}")

            if metrics["f1_score"] > best_f1:
                best_f1               = metrics["f1_score"]
                self.best_model_name  = key
                self.best_pipeline    = pipeline

        print("\n" + "=" * 60)
        print(f"🏆 Best Model: {self.results[self.best_model_name]['name']}")
        print(f"   F1 Score: {self.results[self.best_model_name]['metrics']['f1_score']:.4f}")
        y_best = self.best_pipeline.predict(X_test)
        print("\n📋 Classification Report (Best Model):")
        print(classification_report(y_test, y_best, target_names=["Ham", "Spam"]))
        return X_test, y_test

    def save_model(self) -> str:
        model_path = os.path.join(MODELS_DIR, "spam_classifier.pkl")
        meta_path  = os.path.join(MODELS_DIR, "model_metadata.json")

        joblib.dump(self.best_pipeline, model_path)
        print(f"\n💾 Model saved: {model_path}")

        metadata = {
            "best_model":      self.best_model_name,
            "best_model_name": self.results[self.best_model_name]["name"],
            "trained_at":      datetime.now().isoformat(),
            "metrics":         self.results[self.best_model_name]["metrics"],
            "all_results": {
                k: {"name": v["name"], "metrics": v["metrics"]}
                for k, v in self.results.items()
            },
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"📄 Metadata saved: {meta_path}")
        return model_path

    def run(self) -> str:
        print("🚀 Starting Email Spam Classifier Training Pipeline\n" + "=" * 60)
        df = self.load_data()
        df = self.preprocess(df)
        self.train_and_evaluate(df)
        return self.save_model()


def train() -> str:
    return SpamModelTrainer().run()


if __name__ == "__main__":
    train()
