import azure.functions as func
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils.db_utils import get_synapse_connection, upload_to_blob
import logging

rating_regr = func.Blueprint()

@rating_regr.timer_trigger(
    schedule="0 0 2 * * *", arg_name="myTimer", run_on_startup=True, use_monitor=False
)
def RatingRegression(myTimer: func.TimerRequest):
    if myTimer.past_due:
        logging.info("Timer was past due!")

    logging.info("⭐ Starting Rating Regression using PCA Components...")

    # --- 1️⃣ Load data from Synapse
    conn = get_synapse_connection()
    query = """
        SELECT
            p.listing_id,
            TRY_CAST(l.rating_overall AS FLOAT) AS rating_overall,
            p.PC1, p.PC2, p.PC3, p.PC4, p.PC5, p.PC6, p.PC7, p.PC8, p.PC9, p.PC10,
            p.PC11, p.PC12, p.PC13, p.PC14, p.PC15, p.PC16, p.PC17, p.PC18, 
            p.PC19, p.PC20, p.PC21
        FROM silver.pca_features AS p
        JOIN silver.reviews_summary AS l
            ON p.listing_id = l.listing_id
        WHERE l.rating_overall IS NOT NULL
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        logging.warning("⚠️ No data retrieved for regression — skipping.")
        return

    # --- 2️⃣ Prepare features & target
    df = df.dropna(subset=["rating_overall"])
    X = df[[c for c in df.columns if c.startswith("PC")]]
    y = df["rating_overall"]

    # --- 3️⃣ Train regression model
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    logging.info(f"✅ Regression complete. R² = {r2:.3f}")

    # --- 4️⃣ Prepare coefficients for output
    coef_df = pd.DataFrame({
        "Component": X.columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)

    coef_df["r2_score"] = r2

    # --- 5️⃣ Save to blob
    upload_to_blob(coef_df, prefix="pca_regression")
    logging.info("💾 Coefficients saved to blob.")
