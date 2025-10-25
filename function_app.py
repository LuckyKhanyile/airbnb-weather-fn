import logging
import os
import io
import datetime
import pandas as pd
import requests
import pyodbc
from azure.storage.blob import ContainerClient
import azure.functions as func


app = func.FunctionApp()

@app.timer_trigger(
    schedule="0 0 12 1 * *",     # run at 12 PM UTC on the 1st of every month
    arg_name="myTimer",
    run_on_startup=True,
    use_monitor=False
)
def GetWeatherData(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.info("Timer was past due!")
    logging.info("Starting monthly weather enrichment job...")

    gps_df = load_coordinates_from_synapse()
    logging.info(f"Loaded {len(gps_df)} GPS coordinates from Synapse")

    start_date = datetime.date.today() - datetime.timedelta(days=30)
    end_date = datetime.date.today()

    all_weather = []
    for _, row in gps_df.iterrows():
        try:
            w = get_weather_history(row.latitude, row.longitude, start_date, end_date)
            if not w.empty:
                w["gps_location_key"] = row.gps_location_key
                all_weather.append(w)
        except Exception as e:
            logging.warning(f"Failed for {row.gps_location_key}: {e}")

    if not all_weather:
        logging.warning("No weather data fetched.")
        return

    weather_df = pd.concat(all_weather, ignore_index=True)
    logging.info(f"Fetched {len(weather_df)} total weather rows")

    blob_path = upload_to_blob(weather_df, container="airbnb-weather", prefix="weather")
    logging.info(f"Uploaded Parquet file to blob: {blob_path}")


# ---------------------------------------------------------------------
# 🔹 SQL AUTHENTICATION (username / password)
# ---------------------------------------------------------------------
def get_synapse_connection():
    """
    Connect to Synapse Serverless SQL pool using SQL authentication.
    Expects the following environment variables:
        SYNAPSE_SERVER   e.g. "airbnb-wkp-ondemand.sql.azuresynapse.net"
        SYNAPSE_DATABASE e.g. "airbnb_olap"
        SYNAPSE_USER     e.g. "sqladminuser"
        SYNAPSE_PASSWORD e.g. "StrongPassword!"
    """
    server = os.environ["SYNAPSE_SERVER"]
    database = os.environ["SYNAPSE_DATABASE"]
    user = os.environ["SYNAPSE_USER"]
    password = os.environ["SYNAPSE_PASSWORD"]

    conn_str = (
        "Driver={ODBC Driver 18 for SQL Server};"
        f"Server=tcp:{server},1433;"
        f"Database={database};"
        f"Uid={user};Pwd={password};"
        "Encrypt=yes;TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )

    conn = pyodbc.connect(conn_str)
    return conn


def load_coordinates_from_synapse():
    """Reads distinct GPS coordinates from Synapse silver.location view."""
    query = """
    SELECT TOP 10 
        gps_location_key,
        latitude_round AS latitude,
        longitude_round AS longitude
    FROM silver.location
    WHERE latitude_round IS NOT NULL
    AND longitude_round IS NOT NULL;
    """
    with get_synapse_connection() as conn:
        df = pd.read_sql(query, conn)
    return df


# ---------------------------------------------------------------------
# 🔹 Fetch weather data
# ---------------------------------------------------------------------
def get_weather_history(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "windspeed_10m_max",
        ],
        "timezone": "Africa/Johannesburg",
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json().get("daily", {})
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["latitude"], df["longitude"] = lat, lon
    return df


# ---------------------------------------------------------------------
# 🔹 Upload to Blob Storage
# ---------------------------------------------------------------------
def upload_to_blob(df: pd.DataFrame, container: str, prefix: str = "weather") -> str:
    """Write DataFrame as Parquet into Azure Blob Storage."""
    sas_url = os.environ["BLOB_SAS_URL"]  # container-level SAS URL
    container_client = ContainerClient.from_container_url(sas_url)

    today = datetime.date.today()
    blob_path = f"{prefix}/{today.year}/{today.month:02d}/weather_{today}.parquet"

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(buffer, overwrite=True)
    return blob_path
