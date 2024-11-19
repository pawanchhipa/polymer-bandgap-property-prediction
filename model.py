from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

class PolymerPropertyPredictor:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.feature_selector = None
        
    def optimize_hyperparameters(self, X, y):
        """Perform grid search for hyperparameter optimization"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        self.model = RandomForestRegressor(**grid_search.best_params_)
        return grid_search.best_params_
    
    def select_features(self, X, y, min_features=100):
        """Perform recursive feature elimination"""
        self.feature_selector = RFECV(
            estimator=self.model,
            step=50,
            cv=3,
            min_features_to_select=min_features,
            n_jobs=-1
        )
        self.feature_selector.fit(X, y)
        return X.columns[self.feature_selector.support_]
    
    def cross_validate(self, X, y, n_splits=5):
        """Perform cross-validation"""
        results = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(X_train, y_train)
            y_pred_val = self.model.predict(X_val)
            y_pred_train = self.model.predict(X_train)
            
            results.append({
                'fold': fold + 1,
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'train_r2': r2_score(y_train, y_pred_train),
                'val_r2': r2_score(y_val, y_pred_val)
            })
            
        return pd.DataFrame(results)