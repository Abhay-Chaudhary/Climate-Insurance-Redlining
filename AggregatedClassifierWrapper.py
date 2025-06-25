# by Steven Gubkin
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

class AggregatedClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model):
        self.base_model = base_model

    def fit(self, X, population, percent_class1, **kwargs):
        """
        X: pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        population: array-like of shape (n_samples,)
        percent_class1: array-like of shape (n_samples,)
        kwargs: passed to underlying model's fit method
        """
        X = pd.DataFrame(X)
        population = np.asarray(population)
        percent_class1 = np.asarray(percent_class1)

        # # Compute number of class 1 and class 0 samples per row
        n_class1 = percent_class1 * population
        n_class0 = (1 - percent_class1) * population

        # Remove any rows with 0 total population
        mask = (population > 0)
        X = X.loc[mask].reset_index(drop=True)
        n_class1 = n_class1[mask]
        n_class0 = n_class0[mask]

        # Duplicate each row for class 0 and class 1
        X_aug = pd.concat([X, X], ignore_index=True)
        y_aug = np.array([0]*len(X) + [1]*len(X))
        sample_weight = np.concatenate([n_class0, n_class1])

        # Fit a clone of the model to avoid mutating input
        self.model_ = clone(self.base_model)
        self.model_.fit(X_aug, y_aug, sample_weight=sample_weight, **kwargs)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)