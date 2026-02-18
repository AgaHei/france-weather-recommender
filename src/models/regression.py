"""
regression.py
-------------
Regression model to predict comfort scores from weather features.

Takes 3-day rolling weather features (temp, precipitation, wind) and
predicts a comfort score (0-100) that represents how pleasant the
weekend conditions will be.

Models tested:
- Ridge Regression (baseline, linear)
- Random Forest (non-linear, handles interactions)
- Gradient Boosting (production choice for best accuracy)

The comfort_score labels are computed by our hand-designed formula
in engineer.py, so this is a supervised learning task.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from datetime import datetime

from src.features.engineer import REGRESSION_FEATURES, REGRESSION_TARGET


class ComfortScoreModel:
    """
    Regression model to predict weekend comfort scores.
    
    Supports multiple model types:
    - 'ridge': Ridge regression (fast, interpretable)
    - 'random_forest': Random forest (non-linear, robust)
    - 'gradient_boosting': XGBoost-like (best accuracy)
    """
    
    def __init__(self, model_type: str = 'gradient_boosting', random_state: int = 42):
        """
        Initialize the regression model.
        
        Args:
            model_type: 'ridge', 'random_forest', or 'gradient_boosting'
            random_state: For reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = self._create_model()
        self.feature_names = REGRESSION_FEATURES
        self.target_name = REGRESSION_TARGET
        self.trained_at = None
        self.metrics = {}
        
    def _create_model(self):
        """Create the sklearn model based on model_type."""
        if self.model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=self.random_state)
        
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                random_state=self.random_state
            )
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> dict:
        """
        Train the regression model with train/test split.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (comfort scores)
            test_size: Fraction of data to hold out for testing
        
        Returns:
            dict with training metrics
        """
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features (important for Ridge, doesn't hurt trees)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.trained_at = datetime.now()
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Compute metrics
        self.metrics = {
            'model_type': self.model_type,
            'n_samples': X.shape[0],
            'n_train': len(X_train),
            'n_test': len(X_test),
            
            # Training metrics
            'train_rmse': round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2),
            'train_mae': round(mean_absolute_error(y_train, y_train_pred), 2),
            'train_r2': round(r2_score(y_train, y_train_pred), 3),
            
            # Test metrics (this is what matters!)
            'test_rmse': round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2),
            'test_mae': round(mean_absolute_error(y_test, y_test_pred), 2),
            'test_r2': round(r2_score(y_test, y_test_pred), 3),
            
            'trained_at': self.trained_at.isoformat(),
        }
        
        # Feature importances (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.metrics['feature_importances'] = {
                name: round(float(imp), 3)
                for name, imp in zip(self.feature_names, importances)
            }
        
        # Cross-validation score (optional, more robust estimate)
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=5, scoring='r2', n_jobs=-1
        )
        self.metrics['cv_r2_mean'] = round(float(cv_scores.mean()), 3)
        self.metrics['cv_r2_std'] = round(float(cv_scores.std()), 3)
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict comfort scores for new weather data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Predicted comfort scores (0-100)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Clip to valid range [0, 100]
        return np.clip(predictions, 0, 100)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model on a held-out dataset.
        
        Args:
            X: Feature matrix
            y: True comfort scores
        
        Returns:
            dict with evaluation metrics
        """
        y_pred = self.predict(X)
        
        return {
            'rmse': round(np.sqrt(mean_squared_error(y, y_pred)), 2),
            'mae': round(mean_absolute_error(y, y_pred), 2),
            'r2': round(r2_score(y, y_pred), 3),
            'n_samples': len(y),
        }
    
    def save(self, filepath: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'trained_at': self.trained_at,
            'metrics': self.metrics,
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.trained_at = model_data['trained_at']
        instance.metrics = model_data['metrics']
        
        print(f"📂 Model loaded from {filepath}")
        print(f"   Model type: {instance.model_type}")
        print(f"   Trained at: {instance.trained_at}")
        print(f"   Test R²: {instance.metrics.get('test_r2', 'N/A')}")
        
        return instance


def compare_models(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> pd.DataFrame:
    """
    Train all model types and compare performance.
    
    Args:
        X: Feature matrix
        y: Target values
        test_size: Fraction for testing
    
    Returns:
        DataFrame with comparison metrics
    """
    model_types = ['ridge', 'random_forest', 'gradient_boosting']
    results = []
    
    for model_type in model_types:
        print(f"\n🔧 Training {model_type}...")
        model = ComfortScoreModel(model_type=model_type)
        metrics = model.train(X, y, test_size=test_size)
        
        results.append({
            'model': model_type,
            'test_rmse': metrics['test_rmse'],
            'test_mae': metrics['test_mae'],
            'test_r2': metrics['test_r2'],
            'cv_r2': metrics['cv_r2_mean'],
        })
    
    df = pd.DataFrame(results).sort_values('test_r2', ascending=False)
    
    print("\n📊 Model Comparison:")
    print(df.to_string(index=False))
    
    return df


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing ComfortScoreModel with synthetic data...\n")
    
    np.random.seed(42)
    
    # Generate synthetic features (temp, precip, wind)
    n_samples = 500
    X = np.random.randn(n_samples, 3) * [5, 3, 10] + [15, 5, 20]
    
    # Generate synthetic comfort scores (non-linear relationship)
    # High score when: temp ~20, low precip, low wind
    y = (
        50 * np.exp(-0.5 * ((X[:, 0] - 20) / 6) ** 2) +  # temp component
        30 * np.exp(-X[:, 1] / 5) +                      # precip component
        20 * np.exp(-X[:, 2] / 25) +                     # wind component
        np.random.randn(n_samples) * 3                   # noise
    )
    y = np.clip(y, 0, 100)
    
    # Train a single model
    print("Training Gradient Boosting model...")
    model = ComfortScoreModel(model_type='gradient_boosting')
    metrics = model.train(X, y, test_size=0.2)
    
    print(f"\n✅ Training complete!")
    print(f"   Test RMSE: {metrics['test_rmse']}")
    print(f"   Test MAE: {metrics['test_mae']}")
    print(f"   Test R²: {metrics['test_r2']}")
    print(f"   CV R²: {metrics['cv_r2_mean']} ± {metrics['cv_r2_std']}")
    
    if 'feature_importances' in metrics:
        print(f"\n📊 Feature importances:")
        for feat, imp in metrics['feature_importances'].items():
            print(f"   {feat}: {imp}")
    
    # Test predictions
    X_new = np.array([[20, 0, 10], [10, 30, 50], [25, 5, 15]])
    preds = model.predict(X_new)
    
    print(f"\n🔮 Sample predictions:")
    for i, pred in enumerate(preds):
        print(f"   {X_new[i]} → {pred:.1f}/100")
