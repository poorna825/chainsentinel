"""
Models for the illicit/licit classification task.

Both models use class_weight="balanced": illicit transactions are a small
minority of the training set, so without this they'd mostly just learn to
predict licit, like the dumb baseline.
"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data import feature_columns


def train_baseline(train_df):
    """Fit a logistic regression on f1..f165, weighted for class imbalance."""
    X_train = train_df[feature_columns(train_df)]
    y_train = train_df["label"]
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(train_df, n_estimators=200, random_state=0):
    """Fit a random forest on f1..f165, weighted for class imbalance.

    Not hyperparameter-tuned yet - the point is comparing model families
    against the logistic regression baseline, not squeezing out best settings.
    """
    X_train = train_df[feature_columns(train_df)]
    y_train = train_df["label"]
    model = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced", random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def feature_importance(model, feature_names, top_n=15):
    """Top features by mean impurity reduction across the forest's trees.

    A useful first look, not a rigorous explanation: correlated features
    split credit between them, and the method favors high-variance features.
    """
    pairs = list(zip(feature_names, model.feature_importances_))
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs[:top_n]


def save_model(model, path):
    """Serialize a fitted model to disk so it can be reused without retraining."""
    joblib.dump(model, path)


def load_model(path):
    """Load a model saved by save_model."""
    return joblib.load(path)
