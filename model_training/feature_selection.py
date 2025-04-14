import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, percentile=75):
        self.percentile = percentile
        self.important_indices_ = None
        self.feature_importances_ = None
        self.selected_features_ = None  

    def fit(self, X, y, feature_names=None):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        self.feature_importances_ = rf.feature_importances_
        threshold = np.percentile(self.feature_importances_, self.percentile)
        self.important_indices_ = np.where(self.feature_importances_ >= threshold)[0]

        if feature_names is not None:
            self.selected_features_ = np.array(feature_names)[self.important_indices_]
            print("✅ Selected Important Features:")
            for feat in self.selected_features_:
                print(f"- {feat}")
        return self

    def transform(self, X):
        return X[:, self.important_indices_]

    def plot_feature_importance(self, feature_names):
        sorted_indices = np.argsort(self.feature_importances_)
        sorted_importances = self.feature_importances_[sorted_indices]
        sorted_feature_names = np.array(feature_names)[sorted_indices]

        plt.figure(figsize=(20, 12))
        sns.barplot(y=sorted_importances, x=sorted_feature_names, palette="coolwarm")
        plt.title("Feature Importance", fontsize=20)
        plt.xlabel("Feature Names", fontsize=18)
        plt.xticks(rotation=90, fontsize=12)
        plt.show()
