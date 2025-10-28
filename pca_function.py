import azure.functions as func
import os
from utils.db_utils import get_synapse_connection, upload_to_blob
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
bp_pca = func.Blueprint()

@bp_pca.timer_trigger(
    schedule="0 0 2 * * *", arg_name="myTimer", run_on_startup=True, use_monitor=False
)
def RunPcaAnalysis(myTimer: func.TimerRequest):
    if myTimer.past_due:
        logging.info("Timer was past due!")

    logging.info("📊 Running PCA Analysis...")

    conn = get_synapse_connection()
    df = pd.read_sql("SELECT * FROM silver.listing_features_pca", conn)

    numeric_df = df.select_dtypes(include=["number"]).dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    pca = PCA(n_components=5)
    pcs = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(5)])
    pca_df["listing_id"] = df["listing_id"].iloc[:len(pca_df)]

    upload_to_blob(pca_df, container="airbnb-weather", prefix="pca")
    logging.info("✅ PCA results uploaded successfully.")
