import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

def plot_parity(y_true, y_pred, title="", ylim=None, ax=None):
    """Enhanced parity plot with better styling"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Plot points
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Plot perfect prediction line
    if ylim is None:
        ylim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(ylim, ylim, '--k', label='Perfect prediction')
    
    # Add metrics
    ax.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nR²: {r2:.2f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    
    return ax

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance from model"""
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importances.head(top_n), x='importance', y='feature')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    
    return importances