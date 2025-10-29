import azure.functions as func
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.db_utils import get_synapse_connection, upload_to_blob
import logging


rating_multi_regr = func.Blueprint()

@rating_multi_regr.timer_trigger(
    schedule="0 0 3 * * *", arg_name="myTimer", run_on_startup=True, use_monitor=False
)
def RatingMultiRegression(myTimer: func.TimerRequest):
    """
    Multi-dimensional regression across multiple rating types (accuracy, cleanliness, etc.)
    using PCA components as predictors.
    Saves the R² per rating and coefficient ranking to blob storage.
    """
    if myTimer.past_due:
        logging.info("Timer was past due!")

    logging.info("🚀 Starting Multi-Rating Regression using PCA Components...")

    # --- 1️⃣ Load data from Synapse
    conn = get_synapse_connection()
    query = """
        SELECT
            p.listing_id,
            TRY_CAST(l.review_scores_rating AS FLOAT) AS rating_overall,
            TRY_CAST(l.review_scores_accuracy AS FLOAT) AS rating_accuracy,
            TRY_CAST(l.review_scores_cleanliness AS FLOAT) AS rating_cleanliness,
            TRY_CAST(l.review_scores_checkin AS FLOAT) AS rating_checkin,
            TRY_CAST(l.review_scores_communication AS FLOAT) AS rating_communication,
            TRY_CAST(l.review_scores_location AS FLOAT) AS rating_location,
            TRY_CAST(l.review_scores_value AS FLOAT) AS rating_value,
            p.PC1, p.PC2, p.PC3, p.PC4, p.PC5, p.PC6, p.PC7, p.PC8, p.PC9, p.PC10,
            p.PC11, p.PC12, p.PC13, p.PC14, p.PC15, p.PC16, p.PC17, p.PC18, p.PC19, p.PC20, p.PC21
        FROM silver.pca_features AS p
        JOIN bronze.airbnb_listings AS l
            ON p.listing_id = TRY_CAST(l.id AS BIGINT)
        WHERE l.review_scores_rating IS NOT NULL
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        logging.warning("⚠️ No data retrieved for multi-regression — skipping.")
        return

    # --- 2️⃣ Define targets and features
    targets = [
        "rating_overall", "rating_accuracy", "rating_cleanliness",
        "rating_checkin", "rating_communication",
        "rating_location", "rating_value"
    ]
    X = df[[c for c in df.columns if c.startswith("PC")]]

    # --- 3️⃣ Loop over each target rating
    results = []

    for target in targets:
        df_target = df.dropna(subset=[target])
        if df_target.empty:
            continue

        y = df_target[target]
        model = LinearRegression()
        model.fit(X.loc[df_target.index], y)
        r2 = model.score(X.loc[df_target.index], y)

        top_idx = np.argmax(np.abs(model.coef_))
        results.append({
            "Rating_Type": target,
            "R2_Score": r2,
            "Top_PC": X.columns[top_idx],
            "Top_Coefficient": float(model.coef_[top_idx])
        })

        logging.info(f"✅ {target}: R²={r2:.3f}, Top={X.columns[top_idx]}")

    # --- 4️⃣ Create results DataFrame
    results_df = pd.DataFrame(results).sort_values(by="R2_Score", ascending=False)

    # --- 5️⃣ Save to blob
    upload_to_blob(results_df, prefix="pca_multi_regression")

    logging.info("💾 Multi-rating regression results saved to blob.")
    logging.info("🏁 Completed multi-rating regression successfully.")
