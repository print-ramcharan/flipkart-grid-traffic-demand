import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class SpatialTemporalEngineer:
    def __init__(self, target_col='demand', n_splits=5):
        self.target_col = target_col
        self.n_splits = n_splits
        
        self.level_1_map = {} 
        self.level_2_map = {} 
        self.level_3_map = {} 
        self.global_mean = 0
        
    def _get_hierarchical_mean(self, df, geohash_col, l1_map, l2_map, l3_map, global_mean):
        val = df[geohash_col].map(l1_map)
        val = val.fillna(df[geohash_col].str[:5].map(l2_map))
        val = val.fillna(df[geohash_col].str[:4].map(l3_map))
        return val.fillna(global_mean)

    def fit_transform(self, train_df):
        print("Fitting Pure SpatialTemporalEngineer (No Lags)...")
        df = train_df.copy()
        
        df['geohash_5'] = df['geohash'].str[:5]
        df['geohash_4'] = df['geohash'].str[:4]
        
        # Unseen geohashes are dead zones; their demand is virtually zero.
        self.global_mean = 0.0001 
        self.level_1_map = df.groupby('geohash')[self.target_col].mean().to_dict()
        self.level_2_map = df.groupby('geohash_5')[self.target_col].mean().to_dict()
        self.level_3_map = df.groupby('geohash_4')[self.target_col].mean().to_dict()

        # OOF Encoding
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        df['spatial_prior'] = np.nan
        
        for train_idx, val_idx in kf.split(df):
            X_tr, X_val = df.iloc[train_idx], df.iloc[val_idx]
            l1 = X_tr.groupby('geohash')[self.target_col].mean().to_dict()
            l2 = X_tr.groupby('geohash_5')[self.target_col].mean().to_dict()
            l3 = X_tr.groupby('geohash_4')[self.target_col].mean().to_dict()
            fold_mean = X_tr[self.target_col].mean()
            
            df.loc[val_idx, 'spatial_prior'] = self._get_hierarchical_mean(
                X_val, 'geohash', l1, l2, l3, fold_mean
            )

        return df

    def transform(self, test_df):
        print("Transforming unseen data (Pure Spatial)...")
        df = test_df.copy()
        
        df['geohash_5'] = df['geohash'].str[:5]
        df['geohash_4'] = df['geohash'].str[:4]
        
        df['spatial_prior'] = self._get_hierarchical_mean(
            df, 'geohash', self.level_1_map, self.level_2_map, self.level_3_map, self.global_mean
        )
        
        return df

# --- TARGET BOUNDING HELPERS ---

def bound_and_logit(y):
    """Clips demand to avoid log(0) and applies Logit transform."""
    y_clipped = np.clip(y, 0.0001, 0.9999)
    return np.log(y_clipped / (1 - y_clipped))

def sigmoid_inverse(y_pred):
    """Converts model predictions back to strict [0, 1] bounded demand."""
    return 1 / (1 + np.exp(-y_pred))
