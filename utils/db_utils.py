import os
import pyodbc
import io
from azure.storage.blob import ContainerClient
import pandas as pd
from datetime import date


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

    return pyodbc.connect(conn_str)


def upload_to_blob(df: pd.DataFrame, prefix: str):
    """Uploads DataFrame to Azure Blob Storage as a Parquet file."""
    sas_url = os.environ["BLOB_SAS_URL"]  # container-level SAS URL
    container_client = ContainerClient.from_container_url(sas_url)

    blob_path = f"{prefix}/{date.today().year}/{date.today().month:02d}/{prefix}_{date.today()}.parquet"
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(buffer, overwrite=True)
    return blob_path
