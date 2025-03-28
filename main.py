import requests
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import config
from io import StringIO

def extract_data():
    """Extracts data from the CSV URL."""
    response = requests.get(config.CSV_DATA_URL)
    if response.status_code == 200:
        data = StringIO(response.text)
        df = pd.read_csv(data)
        return df
    else:
        raise Exception(f"Failed to fetch data: {response.status_code} - {response.text}")

def transform_data(df):
    """Transforms the data as needed."""
    # Example transformation: Convert 'Timestamp' to datetime
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # Add more transformations as needed
    return df

def create_table_if_not_exists(conn):
    """Creates the Snowflake table if it doesn't exist."""
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {config.SF_DATABASE}.{config.SF_SCHEMA}.{config.SF_TABLE} (
        "Transaction ID" STRING,
        "Amount" FLOAT,
        "Currency" STRING,
        "Payment Method" STRING,
        "Status" STRING,
        "Timestamp" TIMESTAMP,
        "Merchant ID" STRING,
        "Customer ID" STRING,
        "Payment Gateway" STRING,
        "Authorization Code" STRING
    );
    """
    conn.cursor().execute(create_table_query)
    print(f"Table '{config.SF_DATABASE}.{config.SF_SCHEMA}.{config.SF_TABLE}' is ready.")

def load_data(df):
    """Loads the transformed data into Snowflake."""
    conn = snowflake.connector.connect(
        account=config.SF_ACCOUNT,
        user=config.SF_USER,
        password=config.SF_PASSWORD,
        role=config.SF_ROLE,
        warehouse=config.SF_WAREHOUSE,
        database=config.SF_DATABASE,
        schema=config.SF_SCHEMA
    )
    try:
        create_table_if_not_exists(conn)
        success, nchunks, nrows, _ = write_pandas(conn, df, config.SF_TABLE)
        if success:
            print(f"Successfully inserted {nrows} rows into {config.SF_TABLE}.")
        else:
            print("Data insertion failed.")
    finally:
        conn.close()

def main():
    """Main function to execute the ETL process."""
    df = extract_data()
    if not df.empty:
        transformed_data = transform_data(df)
        load_data(transformed_data)
    else:
        print("No data to process.")

if __name__ == "__main__":
    main()
