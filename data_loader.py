import pandas as pd
import numpy as np
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

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = ''.join([c if c.isalpha() or c.isspace() else ' ' for c in text])
    return text


def preprocess_text_batch(texts):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def process_single(text):
        if pd.isna(text):
            return ""

        text = text.lower()
        text = ''.join([c if c.isalpha() or c.isspace() else ' ' for c in text])

        words = text.split()
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]

        return ' '.join(words)

    return [process_single(text) for text in texts]


def load_and_preprocess_data():
    logging.info("Загрузка и предобработка данных...")

    processed_dir = 'data/processed'
    preprocessor_path = "ddm/preprocessor.joblib"

    if (os.path.exists(os.path.join(processed_dir, 'X_train.csv')) and
            os.path.exists(os.path.join(processed_dir, 'X_test.csv')) and
            os.path.exists(os.path.join(processed_dir, 'y_train.csv')) and
            os.path.exists(os.path.join(processed_dir, 'y_test.csv')) and
            os.path.exists(preprocessor_path)):
        logging.info("Обработанные данные уже существуют. Загружаем их...")

        X_train_df = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        X_test_df = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).squeeze()
        y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).squeeze()
        preprocessor = joblib.load(preprocessor_path)

        logging.info("Данные успешно загружены из кэша")
        return X_train_df, X_test_df, y_train, y_test, preprocessor

    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    cols_to_keep = ['ID', 'name', 'category', 'main_category', 'currency',
                    'deadline', 'goal', 'launched', 'pledged', 'state',
                    'backers', 'country', 'usd pledged', 'usd_pledged_real', 'usd_goal_real']

    df = pd.read_csv('ks-projects-201801.csv',
                     parse_dates=['deadline', 'launched'],
                     usecols=cols_to_keep)

    mask = df['state'].isin(['failed', 'successful'])
    df = df[mask].copy()
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

    logging.info("Начинаю обработку текстовых данных...")
    df['name_prep'] = preprocess_text_batch(df['name'].tolist())

    logging.info("Вычисляю дополнительные текстовые признаки...")
    df['name_len'] = df['name_prep'].str.split().str.len()

    logging.info("Вычисляю сентимент...")
    df['name_sent'] = df['name_prep'].apply(
        lambda x: TextBlob(x).sentiment.polarity if x.strip() != '' else 0
    )

    logging.info("Создаю признаки на основе истории...")
    df['creator_id'] = df['main_category'] + '_' + df['country']
    df = df.sort_values('launch_year')

    group = df.groupby('creator_id')
    df['creator_projects_count'] = group['success'].transform('count')
    df['creator_success_rate'] = group['success'].transform('mean')
    df['creator_avg_goal'] = group['usd_goal_real'].transform('mean')

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
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50))
    ])

    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2)))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_feature)
        ],
        sparse_threshold=0.3,
        n_jobs=-1
    )

    test_size = params['data_processing']['test_size']
    random_state = params['data_processing']['random_state']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logging.info("Начинаю трансформацию данных...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

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

    from scipy import sparse

    os.makedirs(processed_dir, exist_ok=True)

    sparse.save_npz(os.path.join(processed_dir, 'X_train_sparse.npz'), sparse.csr_matrix(X_train_transformed))
    sparse.save_npz(os.path.join(processed_dir, 'X_test_sparse.npz'), sparse.csr_matrix(X_test_transformed))

    X_train_df = pd.DataFrame(
        X_train_transformed.toarray() if sparse.issparse(X_train_transformed) else X_train_transformed,
        columns=feature_names)
    X_test_df = pd.DataFrame(
        X_test_transformed.toarray() if sparse.issparse(X_test_transformed) else X_test_transformed,
        columns=feature_names)

    X_train_df.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test_df.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)

    os.makedirs('ddm', exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)

    logging.info("Данные успешно обработаны и сохранены")

    return X_train_df, X_test_df, y_train, y_test, preprocessor

if __name__ == "__main__":
    load_and_preprocess_data()