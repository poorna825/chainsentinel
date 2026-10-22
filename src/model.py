"""
Models for the illicit/licit classification task.

Both models use class_weight="balanced": illicit transactions are a small
minority of the training set, so without this they'd mostly just learn to
predict licit, like the dumb baseline.
"""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from data import feature_columns


def train_baseline(train_df):
    """Fit a logistic regression on f1..f165, weighted for class imbalance."""
    X_train = train_df[feature_columns(train_df)]
    y_train = train_df["label"]
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(train_df, feature_names=None, n_estimators=200, random_state=0):
    """Fit a random forest, weighted for class imbalance.

    feature_names defaults to f1..f165 (feature_columns(train_df)); pass an
    explicit list to train on a different feature set (e.g. a subset, or
    f1..f165 plus extra non-"f"-prefixed columns like graph degree features).

    Not hyperparameter-tuned yet - the point is comparing model families
    against the logistic regression baseline, not squeezing out best settings.
    """
    if feature_names is None:
        feature_names = feature_columns(train_df)
    X_train = train_df[feature_names]
    y_train = train_df["label"]
    model = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced", random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def oof_predictions(train_df, feature_names, n_folds=5, random_state=0):
    """Out-of-fold predicted illicit probability for every row in train_df.

    Each row is scored by a model fit on the other folds only, so it never
    saw its own label during training. This matters for collective
    classification (using a node's prediction as a feature for its graph
    neighbors): a plain in-sample prediction would let a row's own label
    leak into a feature derived from its own score.
    """
    X = train_df[feature_names].values
    y = train_df["label"].values
    oof = np.full(len(train_df), np.nan)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, val_idx in kf.split(X):
        model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=random_state
        )
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    return oof


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
