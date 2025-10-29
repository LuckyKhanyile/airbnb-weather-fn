import azure.functions as func
import pandas as pd
import numpy as np
import logging
from utils.db_utils import upload_to_blob, get_synapse_connection


rating_factor = func.Blueprint()

@rating_factor.timer_trigger(
    schedule="0 30 3 * * *",  # runs monthly at 03:30 UTC
    arg_name="myTimer",
    run_on_startup=True,
    use_monitor=False
)
def RatingFactorImportance(myTimer: func.TimerRequest):
    if myTimer.past_due:
        logging.info("Timer was past due!")

    logging.info("⭐ Starting Rating Factor Importance Computation...")

    # --- 1️⃣ Load PCA loadings (feature weights per PC) ---
    try:
        conn = get_synapse_connection()
        pca_loadings_query = """
            SELECT *
            FROM silver.pca_loadings
        """
        loadings_df = pd.read_sql(pca_loadings_query, conn)
        conn.close()
    except Exception as e:
        logging.error(f"❌ Failed to read PCA loadings: {e}")
        return

    if loadings_df.empty:
        logging.warning("⚠️ PCA loadings table is empty.")
        return

    # Ensure PC columns are consistent
    loadings_df = loadings_df.rename(columns={c: c.strip() for c in loadings_df.columns})
    pc_columns = [c for c in loadings_df.columns if c.startswith("PC")]

    # --- 2️⃣ Load regression coefficients (from regression_pca parquet) ---
    try:
        conn = get_synapse_connection()
        coef_query = """
            SELECT *
            FROM silver.rating_pca_coef
        """
        coef_df = pd.read_sql(coef_query, conn)
        conn.close()
    except Exception as e:
        logging.error(f"❌ Failed to read regression coefficients: {e}")
        return

    if coef_df.empty:
        logging.warning("⚠️ Regression coefficients are empty.")
        return

    # --- 3️⃣ Align PCA components & coefficients ---
    coef_map = dict(zip(coef_df["Component"], coef_df["Coefficient"]))

    # Keep only PCs that exist in both
    valid_pcs = [pc for pc in pc_columns if pc in coef_map]

    if not valid_pcs:
        logging.warning("⚠️ No overlapping PCA components between loadings and regression coefficients.")
        return

    # --- 4️⃣ Compute feature influence ---
    # Multiply loading weight of each feature by regression coefficient of its PC
    weighted = loadings_df.copy()
    for pc in valid_pcs:
        weighted[pc] = weighted[pc] * coef_map[pc]

    # Aggregate influence across all PCs
    weighted["influence"] = weighted[valid_pcs].sum(axis=1)
    weighted["abs_influence"] = weighted["influence"].abs()

    # --- 5️⃣ Rank features by absolute influence ---
    importance_df = weighted[["Feature", "influence", "abs_influence"]].sort_values(
        by="abs_influence", ascending=False
    )

    # --- 6️⃣ Save output to Blob ---
    upload_to_blob(importance_df, prefix="rating_factor_importance")

    logging.info("💾 Feature importance successfully saved to blob storage.")
