"""
clustering.py
-------------
K-Means clustering model to group cities by weather profile.

Uses 7-day rolling features (temp, precipitation, wind) to identify
weather "regimes" — e.g., Mediterranean (warm, dry), Atlantic (cool, rainy).

The model outputs:
- Cluster assignments for each city
- Cluster centers (prototypical weather profiles)
- Silhouette score (quality metric)

This is used as a coarse filter: we only recommend cities in "good weather" clusters.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
from datetime import datetime

from src.features.engineer import KMEANS_FEATURES


class WeatherClusterModel:
    """
    K-Means clustering for weather profile identification.
    
    Attributes:
        n_clusters: Number of clusters to find (default 3-4)
        scaler: StandardScaler for feature normalization
        model: Trained KMeans model
        feature_names: List of features used for clustering
    """
    
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Initialize the clustering model.
        
        Args:
            n_clusters: Number of weather regimes to identify
            random_state: For reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.feature_names = KMEANS_FEATURES
        self.trained_at = None
        self.silhouette = None
        
    def train(self, X: np.ndarray, city_names: list[str] = None) -> dict:
        """
        Train the K-Means model on weather features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            city_names: Optional list of city names for interpretability
        
        Returns:
            dict with training metrics and cluster info
        """
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
        
        # Standardize features (important for K-Means!)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-Means
        self.model.fit(X_scaled)
        self.trained_at = datetime.now()
        
        # Compute cluster assignments
        labels = self.model.labels_
        
        # Compute silhouette score (quality metric: -1 to +1, higher = better)
        if len(np.unique(labels)) > 1:
            self.silhouette = silhouette_score(X_scaled, labels)
        else:
            self.silhouette = 0.0
        
        # Compute cluster statistics for interpretability
        cluster_stats = self._compute_cluster_stats(X, labels, city_names)
        
        metrics = {
            'n_clusters': self.n_clusters,
            'silhouette_score': round(self.silhouette, 3),
            'inertia': round(self.model.inertia_, 2),
            'n_samples': X.shape[0],
            'trained_at': self.trained_at.isoformat(),
            'cluster_stats': cluster_stats,
        }
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign cluster labels to new data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Cluster labels (0 to n_clusters-1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def _compute_cluster_stats(self, X: np.ndarray, labels: np.ndarray, 
                                city_names: list[str] = None) -> dict:
        """
        Compute interpretable statistics for each cluster.
        
        Returns a dict mapping cluster_id → stats dict with:
        - size: number of samples
        - avg_temp, avg_precip, avg_wind: mean feature values
        - cities: list of cities in this cluster (if city_names provided)
        """
        stats = {}
        
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            cluster_X = X[mask]
            
            cluster_info = {
                'size': int(mask.sum()),
                'avg_temp': round(float(cluster_X[:, 0].mean()), 1),
                'avg_precip': round(float(cluster_X[:, 1].mean()), 1),
                'avg_wind': round(float(cluster_X[:, 2].mean()), 1),
            }
            
            if city_names is not None:
                cluster_cities = [city_names[i] for i, label in enumerate(labels) if label == cluster_id]
                cluster_info['cities'] = cluster_cities
            
            stats[f'cluster_{cluster_id}'] = cluster_info
        
        return stats
    
    def rank_clusters_by_comfort(self, cluster_stats: dict) -> list[int]:
        """
        Rank clusters from best to worst based on ideal weather conditions.
        
        Heuristic: 
        - Prefer moderate temps (15-22°C)
        - Prefer low precipitation
        - Prefer moderate wind (not too strong)
        
        Returns:
            List of cluster IDs sorted best to worst
        """
        scores = []
        
        for cluster_id in range(self.n_clusters):
            stats = cluster_stats[f'cluster_{cluster_id}']
            
            # Temperature score (Gaussian around 18°C)
            temp_score = np.exp(-0.5 * ((stats['avg_temp'] - 18) / 6) ** 2)
            
            # Precipitation score (lower is better)
            precip_score = np.exp(-stats['avg_precip'] / 10)
            
            # Wind score (lower is better)
            wind_score = np.exp(-stats['avg_wind'] / 20)
            
            # Combined score
            comfort = 50 * temp_score + 30 * precip_score + 20 * wind_score
            scores.append((cluster_id, comfort))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [cluster_id for cluster_id, _ in scores]
    
    def save(self, filepath: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'n_clusters': self.n_clusters,
            'feature_names': self.feature_names,
            'trained_at': self.trained_at,
            'silhouette': self.silhouette,
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        instance = cls(n_clusters=model_data['n_clusters'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.trained_at = model_data['trained_at']
        instance.silhouette = model_data['silhouette']
        
        print(f"📂 Model loaded from {filepath}")
        print(f"   Trained at: {instance.trained_at}")
        print(f"   Silhouette score: {instance.silhouette:.3f}")
        
        return instance


def find_optimal_k(X: np.ndarray, k_range: range = range(2, 7)) -> int:
    """
    Use elbow method to find optimal number of clusters.
    
    Args:
        X: Feature matrix
        k_range: Range of k values to test (default 2-6)
    
    Returns:
        Optimal k (number of clusters)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        
        if k > 1:
            sil = silhouette_score(X_scaled, labels)
            silhouettes.append(sil)
        else:
            silhouettes.append(0)
    
    # Find elbow using second derivative
    inertias = np.array(inertias)
    if len(inertias) >= 3:
        second_derivative = np.diff(inertias, 2)
        elbow_idx = np.argmax(second_derivative) + 2  # +2 because of double diff
        optimal_k = list(k_range)[elbow_idx]
    else:
        # Fallback: choose k with highest silhouette
        optimal_k = list(k_range)[np.argmax(silhouettes)]
    
    print(f"📊 Optimal k analysis:")
    for i, k in enumerate(k_range):
        print(f"   k={k}: inertia={inertias[i]:.1f}, silhouette={silhouettes[i]:.3f}")
    print(f"✅ Recommended k: {optimal_k}")
    
    return optimal_k


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing WeatherClusterModel with synthetic data...\n")
    
    # Create synthetic weather patterns
    np.random.seed(42)
    
    # Mediterranean cluster: warm, dry, calm
    med = np.random.randn(50, 3) * [2, 1, 3] + [20, 2, 10]
    
    # Atlantic cluster: cool, rainy, windy
    atlantic = np.random.randn(50, 3) * [3, 2, 5] + [12, 15, 25]
    
    # Continental cluster: variable temps, moderate rain, moderate wind
    continental = np.random.randn(50, 3) * [5, 1.5, 4] + [15, 8, 18]
    
    X = np.vstack([med, atlantic, continental])
    cities = [f"City_{i}" for i in range(150)]
    
    # Train model
    model = WeatherClusterModel(n_clusters=3)
    metrics = model.train(X, cities[:10])  # Just first 10 cities for demo
    
    print(f"\n✅ Training complete!")
    print(f"   Silhouette score: {metrics['silhouette_score']}")
    print(f"   Inertia: {metrics['inertia']}")
    
    print(f"\n📊 Cluster statistics:")
    for cluster_name, stats in metrics['cluster_stats'].items():
        print(f"\n   {cluster_name}:")
        print(f"      Size: {stats['size']}")
        print(f"      Avg temp: {stats['avg_temp']}°C")
        print(f"      Avg precip: {stats['avg_precip']}mm")
        print(f"      Avg wind: {stats['avg_wind']} km/h")
    
    # Test ranking
    ranked = model.rank_clusters_by_comfort(metrics['cluster_stats'])
    print(f"\n🏆 Clusters ranked by comfort: {ranked}")
