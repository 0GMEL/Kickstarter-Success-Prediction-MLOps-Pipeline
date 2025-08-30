import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ssl
import warnings
import certifi
import logging
import os
import yaml
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=certifi.where())

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_preprocess_data():
    logging.info("Загрузка и предобработка данных...")

    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    df = pd.read_csv('ks-projects-201801.csv', parse_dates=['deadline', 'launched'])

    df = df[df['state'].isin(['failed', 'successful'])]
    df['success'] = (df['state'] == 'successful').astype(int)

    cols_to_drop = ['ID', 'pledged', 'usd pledged', 'state', 'backers']
    df = df.drop(columns=cols_to_drop)

    df['name'] = df['name'].fillna("Unknown")

    df['launch_year'] = df['launched'].dt.year
    df['launch_month'] = df['launched'].dt.month
    df['launch_day'] = df['launched'].dt.day
    df['launch_dayofweek'] = df['launched'].dt.dayofweek
    df['launch_hour'] = df['launched'].dt.hour
    df['duration_days'] = (df['deadline'] - df['launched']).dt.days
    df['log_real'] = np.log1p(df['usd_goal_real'])

    df = df.drop(columns=['launched', 'deadline'])

    def preprocess_text(text):
        if pd.isna(text):
            return ""

        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        words = text.split()
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        return ' '.join(words)

    df['name_prep'] = df['name'].apply(preprocess_text)
    df['name_len'] = df['name_prep'].apply(lambda x: len(x.split()))
    df['name_sent'] = df['name_prep'].apply(
        lambda x: TextBlob(x).sentiment.polarity if x.strip() != '' else 0
    )

    df['creator_id'] = df['main_category'] + '_' + df['country']
    df = df.sort_values('launch_year')
    creator_history = df.groupby('creator_id').agg(
        {
            'success': ['count', 'mean'],
            'usd_goal_real': 'mean'
        }
    ).reset_index()

    creator_history.columns = ['creator_id', 'creator_projects_count',
                               'creator_success_rate', 'creator_avg_goal']

    df = pd.merge(df, creator_history, on='creator_id', how='left')

    df = df.drop(columns=['name', 'creator_id', 'usd_pledged_real', 'goal'])

    X = df.drop(columns=['success'])
    y = df['success']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = [col for col in X.select_dtypes(include=['object']).columns if col != 'name_prep']
    text_feature = 'name_prep'

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_feature)
        ],
        sparse_threshold=0.3
    )

    test_size = params['data_processing']['test_size']
    random_state = params['data_processing']['random_state']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if hasattr(X_train_transformed, 'toarray'):
        X_train_transformed = X_train_transformed.toarray()
    if hasattr(X_test_transformed, 'toarray'):
        X_test_transformed = X_test_transformed.toarray()

    feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(features)
        elif name == 'cat':
            cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(features)
            feature_names.extend(cat_features)
        elif name == 'text':
            text_features_out = preprocessor.named_transformers_['text'].named_steps['tfidf'].get_feature_names_out()
            feature_names.extend(text_features_out)

    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    for col in X_train_df.columns:
        if X_train_df[col].dtype == 'object':
            X_train_df[col] = pd.to_numeric(X_train_df[col], errors='coerce')
            X_test_df[col] = pd.to_numeric(X_test_df[col], errors='coerce')

    X_train_df = X_train_df.fillna(0)
    X_test_df = X_test_df.fillna(0)

    os.makedirs('data/processed', exist_ok=True)
    X_train_df.to_csv('data/processed/X_train.csv', index=False)
    X_test_df.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    os.makedirs('ddm', exist_ok=True)
    joblib.dump(preprocessor, "ddm/preprocessor.joblib")

    logging.info("Данные успешно обработаны и сохранены")


if __name__ == "__main__":
    load_and_preprocess_data()