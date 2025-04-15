from typing import Tuple
from sklearn.base import BaseEstimator

def score(text: str, model: BaseEstimator, threshold: float) -> Tuple[bool, float]:
    """
    Scores a single text using a trained sklearn model.

    Args:
        text (str): The input text to classify.
        model (BaseEstimator): A trained sklearn model or pipeline.
        threshold (float): Threshold for classifying as spam.

    Returns:
        Tuple[bool, float]: (prediction: True for spam, False for ham, propensity: spam probability)
    """
    # Predict probability of the "spam" class (assumed to be label 1)
    proba = model.predict_proba([text])[0][1]  # Get probability of class 1 (spam)
    return bool(proba >= threshold), proba
