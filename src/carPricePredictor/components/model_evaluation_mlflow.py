import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse
from carPricePredictor.entity.config_entity import EvaluationConfig
from carPricePredictor.utils.common import create_directories, save_json
import joblib
import os
from carPricePredictor import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import pandas as pd




class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    



    def evaluation(self):

        try:
            # Set our tracking server uri for logging
            tracking_url_type_store= mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

            # Create a new MLflow Experiment
            mlflow.set_experiment("Car Price Predictor")

            X_test_files = os.listdir(self.config.source_dir_x_test)
            if X_test_files:           
                X_test_file = os.path.join(self.config.source_dir_x_test, X_test_files[0])
                X_test = pd.read_csv(X_test_file)
                logger.info("X_test was successfully read by pandas")
            else:
                logger.error("No CSV file found.")
                return None

            

            y_test_files = os.listdir(self.config.source_dir_y_test)
            if y_test_files:           
                y_test_file = os.path.join(self.config.source_dir_y_test, y_test_files[0])
                y_test = pd.read_csv(y_test_file)
                logger.info("y_test was successfully read by pandas")
            else:
                logger.error("No CSV file found.")
                return None

            def load_model():
                model_path = "model/random_forest_model.pkl"
                return joblib.load(model_path)


            def load_params(self):
                params_file_path = os.path.join(self.config.source_dir_params, "best_params.pkl")
                return joblib.load(params_file_path)


            grid_search = load_model()        
            params = load_params(self)


            # Make predictions
            y_pred = grid_search.predict(X_test)
           

            # Evaluate the model
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            score = round(grid_search.score(X_test, y_test) * 100, 2)

            # Log evaluation metrics
            logger.info(f"Mean Absolute Error: {mae}")
            logger.info(f"Mean Squared Error: {mse}")
            logger.info(f"R^2 Score: {r2}")
            logger.info("test score: %s%%", score)

            
            with mlflow.start_run():
                # Log the hyperparameters
                mlflow.log_params(params)

                # Log the metrics
                mlflow.log_metric("score", score)
                mlflow.log_metric("R_squared_Score", r2)
                mlflow.log_metric("Mean Squared Error", mse)
                mlflow.log_metric("Mean Absolute Error", mae)


                # Set a tag that we can use to remind ourselves what this run was for
                mlflow.set_tag("Training Info", "Randomforest with CSV search")
                #mlflow.sklearn.log_model(sk_model=grid_search, artifact_path="model", registered_model_name="random_forest_model")

        
        except Exception as e:
            logger.exception("Error occurred during evaluation:")
            raise 



