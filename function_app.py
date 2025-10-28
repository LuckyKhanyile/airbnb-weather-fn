import azure.functions as func
from weather_function import bp_weather
from sentiment_function import bp_sentiment
from pca_function import bp_pca

# Global FunctionApp instance (used by all other modules)
app = func.FunctionApp()
app.register_functions(bp_weather)
app.register_functions(bp_sentiment)
app.register_functions(bp_pca)