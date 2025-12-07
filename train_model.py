import os
import time
import logging
import yaml
import joblib

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def train_and_evaluate():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')

    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment('Kickstarter Success Prediction')
    experiments = params.get('experiments', {})

    for exp_name, exp_params in experiments.items():
        safe_name = exp_name.replace(' ', '_').lower()
        local_models_dir = "models"
        os.makedirs(local_models_dir, exist_ok=True)
        local_model_path = os.path.join(local_models_dir, f"{safe_name}.pkl")

        with mlflow.start_run(run_name=exp_name):
            logging.info(f"Запуск эксперимента: {exp_name}")
            mlflow.log_params(exp_params)
            mlflow.log_param('model_type', exp_params.get('model_type'))

            if exp_params['model_type'] == 'RandomForest':
                model = RandomForestClassifier(
                    n_estimators=exp_params.get('n_estimators', 100),
                    max_depth=exp_params.get('max_depth', None),
                    min_samples_split=exp_params.get('min_samples_split', 2),
                    min_samples_leaf=exp_params.get('min_samples_leaf', 1),
                    class_weight='balanced' if exp_params.get('class_weight') else None,
                    random_state=exp_params.get('random_state', 42),
                    n_jobs=-1
                )
            elif exp_params['model_type'] == 'GradientBoosting':
                model = GradientBoostingClassifier(
                    n_estimators=exp_params.get('n_estimators', 100),
                    learning_rate=exp_params.get('learning_rate', 0.1),
                    max_depth=exp_params.get('max_depth', 3),
                    subsample=exp_params.get('subsample', 1.0),
                    random_state=exp_params.get('random_state', 42)
                )
            else:
                raise ValueError(f"Неизвестный тип модели: {exp_params['model_type']}")

            if os.path.exists(local_model_path):
                model = joblib.load(local_model_path)
                logging.info(f"Модель {exp_name} загружена из кэша: {local_model_path}")
                training_time = 0.0
            else:
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                joblib.dump(model, local_model_path)
                logging.info(f"Модель {exp_name} обучена и сохранена локально: {local_model_path}")

            y_pred = model.predict(X_test)
            y_pred_proba = None
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("training_time", training_time)

            if y_pred_proba is not None:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                mlflow.log_metric("roc_auc", roc_auc)
                logging.info(f"ROC-AUC: {roc_auc:.4f}")
            else:
                logging.warning("Модель не поддерживает predict_proba — ROC-AUC пропущен.")

            mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.log_param("feature_count", X_train.shape[1])
            mlflow.log_param("training_samples", X_train.shape[0])
            mlflow.log_param("test_samples", X_test.shape[0])

            logging.info(f"Эксперимент {exp_name} завершен. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    train_and_evaluate()