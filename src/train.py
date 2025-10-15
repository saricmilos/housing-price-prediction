# train.py
from src.data_loader import load_datasets
from src.preprocess import Preprocessor

from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor

import joblib
import time
import os

def train_models(n_iter_search=30, random_state=42, save_models=True, models_folder="models"):
    # -------------------------------
    # Load datasets
    # -------------------------------
    datasets = load_datasets()
    train_df = datasets["train"]
    test_df = datasets["test"]

    # -------------------------------
    # Preprocess
    # -------------------------------
    pre = Preprocessor()
    train_processed = pre.fit_transform(train_df, target_col="SalePrice")
    X_train = train_processed.drop(columns=["SalePrice"])
    y_train = train_processed["SalePrice"]

    X_test = pre.transform(test_df)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # -------------------------------
    # Define models and hyperparameters
    # -------------------------------
    models_and_grids = {
        'XGB': {
            'model': XGBRegressor(random_state=random_state, n_jobs=-1, verbosity=0),
            'params': {
                'model__n_estimators': [100, 300, 700],
                'model__max_depth': [3, 6, 9],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__subsample': [0.6, 0.8, 1.0],
            }
        },
        'RFR': {
            'model': RandomForestRegressor(random_state=random_state, n_jobs=-1),
            'params': {
                'model__n_estimators': [100, 300],
                'model__max_depth': [None, 10, 30],
                'model__min_samples_leaf': [1, 3, 5]
            }
        },
        'GBR': {
            'model': GradientBoostingRegressor(random_state=random_state),
            'params': {
                'model__n_estimators': [100, 300],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [3, 5, 8]
            }
        },
        'LR': {
            'model': ElasticNet(random_state=random_state),
            'params': {
                'model__alpha': [0.1, 1.0, 10.0],
                'model__l1_ratio': [0.1, 0.5, 0.9]
            }
        }
    }

    # -------------------------------
    # Cross-validation and search
    # -------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    best_models = {}
    tuning_summary = []

    # Ensure models folder exists
    os.makedirs(models_folder, exist_ok=True)

    for name, mg in models_and_grids.items():
        print(f"\n--- Tuning {name} ---")
        start = time.time()

        # Pipeline: just the model (data already numeric)
        pipeline = Pipeline([
            ('model', mg['model'])
        ])

        param_dist = mg.get('params', {})
        rs = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            scoring='neg_root_mean_squared_error',
            cv=kf,
            verbose=1,
            n_jobs=-1,
            random_state=random_state,
            refit=True
        )

        rs.fit(X_train, y_train)
        elapsed = time.time() - start

        best_models[name] = rs.best_estimator_
        tuning_summary.append({
            'model': name,
            'best_score_negRMSE': rs.best_score_,
            'best_RMSE': -rs.best_score_,
            'best_params': rs.best_params_,
            'n_iter': n_iter_search,
            'time_sec': elapsed
        })

        print(f"{name} done in {elapsed:.0f}s — best RMSE: {-rs.best_score_:.4f}")

        # Optionally save model
        if save_models:
            joblib.dump(rs.best_estimator_, f"{models_folder}/{name}_best_model.pkl")

    return best_models, tuning_summary, X_train, y_train, X_test

# -------------------------------
# Optional: run when called directly
# -------------------------------
if __name__ == "__main__":
    best_models, tuning_summary, X_train, y_train, X_test = train_models()
    print("\nAll models trained and saved!")