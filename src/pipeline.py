import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from data_loader import load_and_preprocess
from features import SpatialTemporalEngineer, bound_and_logit, sigmoid_inverse

def main():
    print("Loading raw dataset...")
    train = load_and_preprocess("dataset/train.csv")
    test = load_and_preprocess("dataset/test.csv")
    
    print("\n[Holdout Validation] Splitting Day 48 vs Day 49")
    train_fold = train[train['day'] == 48].copy()
    val_fold = train[train['day'] == 49].copy()
    
    train_fold['demand_logit'] = bound_and_logit(train_fold['demand'])
    val_fold['demand_logit'] = bound_and_logit(val_fold['demand'])
    train['demand_logit'] = bound_and_logit(train['demand'])
    
    # Feature Engineering
    engineer = SpatialTemporalEngineer(target_col='demand', n_splits=5)
    X_train_fold_full = engineer.fit_transform(train_fold)
    X_val_fold_full = engineer.transform(val_fold)
    
    exclude_cols = ['demand', 'demand_logit', 'Index', 'timestamp', 'geohash', 'day']
    features = [c for c in X_train_fold_full.columns if c not in exclude_cols]
    
    cat_features = ['RoadType', 'Weather', 'geohash_p4', 'geohash_4', 'geohash_5', 'RoadType_NumberofLanes', 'LargeVehicles', 'Landmarks']
    cat_features = [c for c in cat_features if c in features]
    cat_features_idx = [features.index(c) for c in cat_features]
    
    # Cast categoricals
    for df_x in [X_train_fold_full, X_val_fold_full, train, test]:
        for c in cat_features:
            if c in df_x.columns:
                df_x[c] = df_x[c].astype(str)
    
    print("\nTraining CatBoost on Validation Fold...")
    model = CatBoostRegressor(
        iterations=300, 
        learning_rate=0.05, 
        depth=6, 
        cat_features=cat_features_idx, 
        verbose=100, 
        random_seed=42
    )
    model.fit(X_train_fold_full[features], X_train_fold_full['demand_logit'])
    
    # Holdout Validation Metrics
    val_logit_preds = model.predict(X_val_fold_full[features])
    val_preds = sigmoid_inverse(val_logit_preds)
    y_val_true = val_fold['demand']
    
    r2 = r2_score(y_val_true, val_preds)
    mae = mean_absolute_error(y_val_true, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val_true, val_preds))
    
    print(f"\n--- Validation Metrics ---")
    print(f"R² Score : {max(0, r2 * 100):.4f}%")
    print(f"MAE      : {mae:.6f}")
    print(f"RMSE     : {rmse:.6f}")
    
    print("\n[Final Inference] Training on Full Dataset...")
    engineer_full = SpatialTemporalEngineer(target_col='demand', n_splits=5)
    X_train_full = engineer_full.fit_transform(train)
    
    # Re-cast categorical for full set since it was rebuilt
    for c in cat_features:
        if c in X_train_full.columns:
            X_train_full[c] = X_train_full[c].astype(str)
            
    full_model = CatBoostRegressor(
        iterations=300, 
        learning_rate=0.05, 
        depth=6, 
        cat_features=cat_features_idx, 
        verbose=100, 
        random_seed=42
    )
    full_model.fit(X_train_full[features], train['demand_logit'])
    
    print("Generating Test Predictions...")
    test['demand'] = np.nan
    X_test_full = engineer_full.transform(test)
    for c in cat_features:
        if c in X_test_full.columns:
            X_test_full[c] = X_test_full[c].astype(str)
            
    test_logit_preds = full_model.predict(X_test_full[features])
    test_preds = sigmoid_inverse(test_logit_preds)
    
    sub = pd.DataFrame({'Index': test['Index'], 'demand': test_preds})
    sub.to_csv('dataset/submission.csv', index=False)
    print("Successfully saved predictions to dataset/submission.csv!")

if __name__ == "__main__":
    main()
