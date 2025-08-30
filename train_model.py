from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, classification_report, roc_curve
from data_loader import *
import mlflow
import mlflow.sklearn
import time
import logging
import pandas as pd
import os

def train_and_evalute():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')

    mlflow.set_experiment('Kickstarter Success Prediction')
    experiments = params['experiments']

    for exp_name, exp_params in experiments.items():
        with mlflow.start_run():
            logging.info(f"Запуск эксперимента: {exp_name}")

            mlflow.set_tracking_uri("mlruns")
            mlflow.log_params(exp_params)
            mlflow.log_param('model_type', exp_params['model_type'])

            if exp_params['model_type'] == 'RandomForest':
                model = RandomForestClassifier(
                    n_estimators = exp_params['n_estimators'],
                    max_depth = exp_params['max_depth'],
                    min_samples_split = exp_params['min_samples_split'],
                    min_samples_leaf = exp_params['min_samples_leaf'],
                    class_weight = 'balanced' if exp_params['class_weight'] else None,
                    random_state = exp_params['random_state'],
                    n_jobs = -1
                )
            elif exp_params['model_type'] == 'GradientBoosting':
                model = GradientBoostingClassifier(
                    n_estimators = exp_params['n_estimators'],
                    learning_rate = exp_params['learning_rate'],
                    max_depth = exp_params['max_depth'],
                    subsample = exp_params['subsample'],
                    random_state = exp_params['random_state']
                )
            else:
                raise ValueError(f"Неизвестный тип модели: {exp_params['model_type']}")

            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)

            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("training_time", training_time)

            mlflow.log_param("feature_count", X_train.shape[1])
            mlflow.log_param("training_samples", X_train.shape[0])
            mlflow.log_param("test_samples", X_test.shape[0])

            mlflow.sklearn.log_model(model, "model")

            model_filename = f"{exp_name.replace(' ', '_').lower()}.pkl"
            model_path = f"models/{model_filename}"
            os.makedirs(os.path.dirname(model_path), exist_ok = True)

            mlflow.sklearn.save_model(model, model_path)

            logging.info(f"Эксперимент {exp_name} завершен. ROC-AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_and_evalute()