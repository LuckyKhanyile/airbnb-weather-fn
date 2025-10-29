
import azure.functions as func
from utils.db_utils import get_synapse_connection, upload_to_blob
import pandas as pd
from textblob import TextBlob
import logging

bp_sentiment = func.Blueprint()
@bp_sentiment.timer_trigger(
    schedule="0 0 1 * * *", arg_name="myTimer", run_on_startup=True, use_monitor=False
)
def GetReviewSentiment(myTimer: func.TimerRequest):
    if myTimer.past_due:
        logging.info("Timer was past due!")

    logging.info("💬 Starting sentiment analysis...")

    conn = get_synapse_connection()
    query = """
        SELECT DISTINCT review_id, listing_id, review_text, review_date
        FROM silver.reviews
        WHERE review_text IS NOT NULL;
    """
    df = pd.read_sql(query, conn)

    df["polarity"] = df["review_text"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["subjectivity"] = df["review_text"].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    upload_to_blob(df,  prefix="sentiment")
    logging.info(f"✅ Sentiment results uploaded: {len(df)} rows")
