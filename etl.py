import logging
import requests
import pyodbc
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from textblob import TextBlob
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Database connection
def create_connection():
    try:
        conn_str = (
            f'DRIVER={config.SQL_DRIVER};'
            f'SERVER={config.SQL_SERVER};'
            f'DATABASE={config.SQL_DATABASE};'
            f'UID={config.SQL_USER};'
            f'PWD={config.SQL_PASSWORD};'
        )
        conn = pyodbc.connect(conn_str)
        logging.info("Successfully connected to the database.")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        return None

# Create table if not exists
def create_table_if_not_exists(conn):
    create_table_query = f'''
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{config.SQL_TABLE}' AND xtype='U')
    CREATE TABLE {config.SQL_TABLE} (
        id INT PRIMARY KEY,
        title NVARCHAR(255),
        price FLOAT,
        description NVARCHAR(1024),
        category NVARCHAR(50),
        image NVARCHAR(255),
        rating_rate FLOAT,
        rating_count INT,
        predicted_price FLOAT,
        sentiment FLOAT,
        cluster INT,
        anomaly INT
    );
    '''
    try:
        with conn.cursor() as cursor:
            cursor.execute(create_table_query)
            conn.commit()
            logging.info(f"Table '{config.SQL_TABLE}' is ready.")
    except Exception as e:
        logging.error(f"Error creating table: {e}")

# Extract data from API
def extract_data(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        logging.info("Data extracted successfully.")
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to retrieve data: {e}")
        return None

# Transform data
def transform_data(data):
    if not data:
        logging.warning("No data to transform.")
        return None

    transformed_data = [
        {
            "id": item["id"],
            "title": item["title"],
            "price": item["price"],
            "description": item.get("description", ""),
            "category": item["category"],
            "image": item["image"],
            "rating_rate": item["rating"]["rate"],
            "rating_count": item["rating"]["count"],
        }
        for item in data
    ]
    df = pd.DataFrame(transformed_data)
    logging.info(f"Data transformed: {df.shape[0]} records ready for insertion.")
    return df

# AI/ML Features

def train_price_prediction_model(df):
    df.fillna(df.median(), inplace=True)
    X = df[['rating_rate', 'rating_count']]
    y = df['price']
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

def predict_price(model, df):
    return model.predict(df[['rating_rate', 'rating_count']])

def recommend_products(df, product_id):
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['description'].fillna(""))
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        idx = df.index[df['id'] == product_id].tolist()[0]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
        return [df.iloc[i[0]]['id'] for i in sim_scores[1:6]]
    except Exception as e:
        logging.error(f"Error in recommendation system: {e}")
        return []

def analyze_sentiment(description):
    return TextBlob(description).sentiment.polarity if description else 0

def cluster_products(df):
    try:
        X = df[['price', 'rating_rate', 'rating_count']]
        kmeans = KMeans(n_clusters=3, random_state=42)
        return kmeans.fit_predict(X)
    except Exception as e:
        logging.error(f"Error clustering products: {e}")
        return None

def detect_anomalies(df):
    try:
        model = IsolationForest()
        return model.fit_predict(df[['price', 'rating_rate', 'rating_count']])
    except Exception as e:
        logging.error(f"Error in anomaly detection: {e}")
        return None

def load_data(conn, df):
    if df is None or df.empty:
        logging.warning("No data to load.")
        return
    try:
        with conn.cursor() as cursor:
            insert_query = f'''
            INSERT INTO {config.SQL_TABLE} (id, title, price, description, category, image, rating_rate, rating_count, predicted_price, sentiment, cluster, anomaly)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            '''
            for _, row in df.iterrows():
                cursor.execute(insert_query, tuple(row))
            conn.commit()
            logging.info(f"Inserted {len(df)} records into '{config.SQL_TABLE}'.")
    except Exception as e:
        logging.error(f"Error inserting data: {e}")

def main():
    url = config.API_URL
    conn = create_connection()
    if conn:
        create_table_if_not_exists(conn)
        data = extract_data(url)
        df = transform_data(data)
        if df is not None and not df.empty:
            model = train_price_prediction_model(df)
            df['predicted_price'] = predict_price(model, df)
            df['sentiment'] = df['description'].apply(analyze_sentiment)
            df['cluster'] = cluster_products(df)
            df['anomaly'] = detect_anomalies(df)
            load_data(conn, df)
        conn.close()
        logging.info("ETL process completed with AI/ML features.")
    else:
        logging.error("ETL process aborted due to database connection failure.")

if __name__ == '__main__':
    main()


