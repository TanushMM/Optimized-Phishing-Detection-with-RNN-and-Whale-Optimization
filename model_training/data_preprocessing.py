import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from feature_selection import FeatureSelector

def load_and_preprocess_data(data_path):
    """ Loads dataset, applies feature selection & scaling """
    data = pd.read_csv(data_path)
    data.dropna(axis=0, inplace=True)
    data.dropna(axis=1, inplace=True)     

    feature_names = data.columns[:-1]
    X = data.iloc[:, :-1].values  
    y = data.iloc[:, -1].values  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_selector = FeatureSelector(percentile=75)
    feature_selector.fit(X_train, y_train, feature_names=feature_names)

    feature_pipeline = Pipeline([
        ("feature_selection", feature_selector),
        ("scaling", MinMaxScaler())
    ])

    X_train = feature_pipeline.fit_transform(X_train, y_train)
    X_test = feature_pipeline.transform(X_test)

    feature_selector.plot_feature_importance(feature_names)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test
