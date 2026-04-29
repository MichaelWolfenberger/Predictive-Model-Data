import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_prep import load_and_clean_data

def train_and_evaluate():
    # 1. Get the cleaned data
    df, features, target = load_and_clean_data('../data/co2_emissions.csv')

    # Drop leaky columns
    leaky_cols = ['coal_co2', 'oil_co2', 'gas_co2', 'cement_co2', 'flaring_co2', 'other_industry_co2']
    features = [f for f in features if f not in leaky_cols]

    X = df[features]
    y = df[target]

    # --- THE CLASSIFICATION PIVOT (Fixing B4) ---
    # Convert the continuous CO2 number into a binary category:
    # 1 for 'High Emitter' (above median), 0 for 'Low Emitter'
    median_co2 = y.median()
    y_class = (y > median_co2).astype(int)

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

    # --- HYPERPARAMETER TUNING (Fixing B6) ---
    print("Running GridSearchCV for Hyperparameter Tuning (This may take a minute)...")
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }

    # We switch to XGBClassifier instead of Regressor
    base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    print(f"Best Parameters Found: {grid_search.best_params_}")

    # --- CROSS-VALIDATION (Fixing B5) ---
    print("\nRunning 5-Fold Cross Validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.2f}")

    # 4. Make Predictions
    predictions = model.predict(X_test)

    # 5. Evaluate Performance (Fixing B4 - Classification Metrics)
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # 6. SHAP Value Analysis
    print("\nGenerating SHAP Summary Plot...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

if __name__ == "__main__":
    train_and_evaluate()