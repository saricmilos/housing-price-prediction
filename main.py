# main.py
from src.train import train_models

best_models, tuning_summary, X_train, y_train, X_test = train_models(
    n_iter_search=30,
    save_models=True
)