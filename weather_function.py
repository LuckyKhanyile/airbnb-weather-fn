import azure.functions as func
import os
from utils.db_utils import get_synapse_connection, upload_to_blob
import pandas as pd
import datetime, requests, logging

bp_weather = func.Blueprint()

@bp_weather.timer_trigger(
    schedule="0 0 12 1 * *", arg_name="myTimer", run_on_startup=False, use_monitor=False
)
def GetWeatherData(myTimer: func.TimerRequest):
    if myTimer.past_due:
        logging.info("Timer was past due!")

    logging.info("🌦 Fetching weather data...")

    conn = get_synapse_connection()
    query = """
    SELECT DISTINCT gps_location_key, latitude_round AS latitude, longitude_round AS longitude
    FROM silver.location
    WHERE latitude_round IS NOT NULL AND longitude_round IS NOT NULL;
    """
    gps_df = pd.read_sql(query, conn)
    start_date = datetime.date.today() - datetime.timedelta(days=30)
    end_date = datetime.date.today()

    all_weather = []
    for _, row in gps_df.iterrows():
        try:
            data = get_weather_history(row.latitude, row.longitude, start_date, end_date)
            if not data.empty:
                data["gps_location_key"] = row.gps_location_key
                all_weather.append(data)
        except Exception as e:
            logging.warning(f"Weather fetch failed for {row.gps_location_key}: {e}")

    if all_weather:
        df = pd.concat(all_weather, ignore_index=True)
        upload_to_blob(df, container="airbnb-weather", prefix="weather")
        logging.info(f"✅ Uploaded {len(df)} weather rows")
    else:
        logging.warning("No weather data collected.")


def get_weather_history(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
        "daily": ["temperature_2m_max","temperature_2m_min","temperature_2m_mean","precipitation_sum","windspeed_10m_max"],
        "timezone": "Africa/Johannesburg"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json().get("daily", {})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["latitude"], df["longitude"] = lat, lon
    return df
