"""HW3 submission module. The autograder imports exactly these names.

The grader loads the data exactly as the starter does (170 float sensor columns with missing values),
fits build_pipeline() on its own training split, flags trucks with predict_proba(...)[:, 1] >= THRESHOLD
on a held-out split you do not see, and computes the cost per truck (10 per check, 500 per missed failure).
"""
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier

THRESHOLD: float = 0.38


def build_pipeline() -> BaseEstimator:
    """Return an UNFITTED classifier (preprocessing included) that implements predict_proba."""
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=0,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate=0.1,
        max_depth=3,
        max_iter=200,
    )
