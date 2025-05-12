# src/model_training.py

import pickle
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore

def train_random_forest(X, y):
    """Train Random Forest model."""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation:\nMSE: {mse:.2f}\nR² Score: {r2:.2f}")

    return model

def save_model(model, filename):
    """Save model to disk."""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    """Load model from disk."""
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model
