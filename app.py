from flask import Flask, request, jsonify
import os
import joblib
import pandas as pd
import numpy as np
import logging
import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/gradientboosting_tuned.pkl/model.pkl")
PREPROCESSOR_PATH = os.environ.get("PREPROCESSOR_PATH", "ddm/preprocessor.joblib")

model = None
preprocessor = None
class_names = None

nltk.data.path.append('/usr/local/share/nltk_data')

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir='/usr/local/share/nltk_data')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='/usr/local/share/nltk_data')

lemmatizer = WordNetLemmatizer()

def load_model(path):
    logger.info("Loading model from: %s", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

def load_preprocessor(path):
    logger.info("Loading preprocessor from: %s", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor file not found: {path}")
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
    preprocessor = load_preprocessor(PREPROCESSOR_PATH)
    class_names = getattr(model, "classes_", None)

    logger.info("Model and preprocessor loaded successfully. class_names: %s", str(class_names is not None))

    categorical_columns = []
    if hasattr(preprocessor, 'named_transformers_'):
        for name in preprocessor.named_transformers_:
            if 'cat' in name.lower():
                cat_pipe = preprocessor.named_transformers_[name]
                if hasattr(cat_pipe, 'named_steps'):
                    if 'imputer' in cat_pipe.named_steps:
                        imputer = cat_pipe.named_steps['imputer']
                        imputer.missing_values = None
                        logger.info("Updated categorical imputer missing_values to None")
                for t_name, t, t_cols in preprocessor.transformers_:
                    if t_name == name:
                        categorical_columns = t_cols
                        break
    logger.info("Categorical columns: %s", categorical_columns)
except Exception as e:
    logger.exception("Failed to load model/preprocessor at startup: %s", e)
    model = None
    preprocessor = None

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if model is not None and preprocessor is not None else "not ready",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or preprocessor is None:
        return jsonify({"error": "Model or preprocessor not loaded"}), 500
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        X_df = pd.DataFrame([data] if isinstance(data, dict) else data)

        X_df['launched'] = pd.to_datetime(X_df['launched'])
        X_df['deadline'] = pd.to_datetime(X_df['deadline'])
        X_df['launch_year'] = X_df['launched'].dt.year
        X_df['launch_month'] = X_df['launched'].dt.month
        X_df['launch_day'] = X_df['launched'].dt.day
        X_df['launch_dayofweek'] = X_df['launched'].dt.dayofweek
        X_df['launch_hour'] = X_df['launched'].dt.hour
        X_df['duration_days'] = (X_df['deadline'] - X_df['launched']).dt.days
        X_df['log_real'] = np.log1p(X_df['usd_goal_real'])
        X_df = X_df.drop(columns=['launched', 'deadline'], errors='ignore')
        X_df['name_prep'] = X_df.get('name', "Unknown").apply(preprocess_text)
        X_df['creator_id'] = X_df.get('main_category', '') + '_' + X_df.get('country', '')
        X_df['creator_projects_count'] = X_df.get('creator_projects_count', 1)
        X_df['creator_success_rate'] = X_df.get('creator_success_rate', 0.5)
        X_df['creator_avg_goal'] = X_df.get('usd_goal_real', 0)

        X_df = X_df.drop(columns=['name', 'creator_id', 'ID', 'backers', 'pledged', 'state'], errors='ignore')
        expected_cols = preprocessor.feature_names_in_
        for col in expected_cols:
            if col not in X_df.columns:
                if col in categorical_columns:
                    X_df[col] = 'Unknown'
                else:
                    X_df[col] = np.nan

        numeric_cols = X_df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)

        categorical_cols = X_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X_df[col] = X_df[col].astype(str).fillna('Unknown')

        X_transformed = preprocessor.transform(X_df)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        pred = model.predict(X_transformed)
        probabilities = model.predict_proba(X_transformed) if hasattr(model, "predict_proba") else None

        response = {
            "prediction": pred.tolist(),
            "probabilities": probabilities.tolist() if probabilities is not None else None
        }

        return jsonify(response)

    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host=os.environ.get("FLASK_HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 5050)))