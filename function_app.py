import azure.functions as func
from weather_function import bp_weather
from sentiment_function import bp_sentiment
from pca_function import bp_pca
from feature_extraction_pca import bp_feature  
from rating_regression_pca import rating_regr 
from rating_factor_importance_pca import rating_factor
from rating_multi_regress_pca import rating_multi_regr

# Global FunctionApp instance (used by all other modules)
app = func.FunctionApp()
app.register_functions(bp_weather)
app.register_functions(bp_sentiment)
app.register_functions(bp_pca)
app.register_functions(bp_feature)   
app.register_functions(rating_regr)   
app.register_functions(rating_factor)
app.register_blueprint(rating_multi_regr)