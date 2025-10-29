import azure.functions as func
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils.db_utils import get_synapse_connection, upload_to_blob


bp_feature = func.Blueprint()

@bp_feature.timer_trigger(
    schedule="0 30 1 * * *",   # 🕐 Run monthly at 01:30 UTC
    arg_name="myTimer",
    run_on_startup=True,
    use_monitor=False
)
def RunFeatureExtraction(myTimer: func.TimerRequest) -> None:
    """Performs PCA feature extraction and saves results to Blob as Parquet."""
    if myTimer.past_due:
        logging.warning("⏰ Timer was past due!")

    logging.info("🚀 Starting PCA feature extraction...")

    # --- 1️⃣ Read input data from Synapse ---------------------------
    query = "SELECT * FROM silver.listing_features_pca"
    with get_synapse_connection() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        logging.warning("⚠️ No data returned from silver.listing_features_pca.")
        return

    logging.info(f"✅ Loaded {len(df)} records for PCA feature extraction")

    coords = df[["latitude", "longitude"]].dropna()
    n_clusters = df["neighbourhood_cleansed"].nunique()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["location_ward"] = kmeans.fit_predict(coords)
    
    # --- 2️⃣ Select numeric columns ---------------------------------
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Remove identifiers or keys
    id_cols = ["listing_id","latitude", "longitude"]  # adjust as needed
    numeric_cols = [col for col in numeric_cols if col not in id_cols]
    
    if not numeric_cols:
        logging.warning("⚠️ No numeric columns found for PCA")
        return

    X = df[numeric_cols].dropna()

    # --- 3️⃣ Standardize -------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
        
    # --- 4️⃣ Run PCA (retain 90% variance) --------------------------
    pca = PCA(n_components=0.9)
    X_pca = pca.fit_transform(X_scaled)
    
        # --- 💾 Compute and store PCA loadings ---
    loadings = pd.DataFrame(
        pca.components_.T,
        index=numeric_cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    ).reset_index().rename(columns={"index": "Feature"})
    upload_to_blob(loadings, prefix="pca_loadings")
    logging.info("✅ PCA loadings uploaded successfully")
    
    
    explained = pca.explained_variance_ratio_.sum()

    logging.info(f"📊 PCA generated {X_pca.shape[1]} components explaining {explained:.2%} variance")

    # --- 5️⃣ Build PCA DataFrame ------------------------------------
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    if "listing_id" in df.columns:
        pca_df["listing_id"] = df["listing_id"].values[:len(pca_df)]

    upload_to_blob(pca_df, prefix="pca_features")
    logging.info(f"✅ PCA feature scores saved to successfully")

    logging.info("🏁 PCA feature extraction completed successfully.")
