import os
import tensorflow as tf
from carPricePredictor.entity.config_entity import TrainingConfig
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.tree import DecisionTreeRegressor
from carPricePredictor import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    
    def train(self):
        try:
            data_files = os.listdir(self.config.root_dir_data)
            if data_files:


                # Load preprocessed data
                csv_file = os.path.join(self.config.root_dir_data, data_files[0])
                df = pd.read_csv(csv_file)

                # Split features and target
                X = df.drop(columns=["sellingprice"])
                y = df["sellingprice"]

                X_train, X_test, y_train, y_test = train_test_split(
                X, 
                y, test_size=0.3, 
                random_state=10
                )

                # Create directories if they don't exist
                os.makedirs(self.config.root_dir_X_test, exist_ok=True)
                os.makedirs(self.config.root_dir_y_test, exist_ok=True)

                # Save X_test and y_test as CSV files
                X_test.to_csv(os.path.join(self.config.root_dir_X_test, "X_test.csv"), index=False)
                y_test.to_csv(os.path.join(self.config.root_dir_y_test, "y_test.csv"), index=False)
                # Dump X_test and y_test
                #X_test_path = os.path.join(self.config.root_dir_X_test, "X_test")
                #joblib.dump(X_test, X_test_path)

                #y_test_path = os.path.join(self.config.root_dir_y_test, "y_test")
                #joblib.dump(y_test, y_test_path)
                



                #Create the parameter grid based on the results of random search 
                # param_grid = {
                #     'bootstrap': [True],
                #     'max_depth': [80, 90, 100, 110],
                #     'max_features': [2, 3],
                #     'min_samples_leaf': [3, 4, 5],
                #     'min_samples_split': [8, 10, 12],
                #     'n_estimators': [100, 200, 300, 1000]
                # }

                param_grid = {
                    'bootstrap': [True],
                    'max_depth': [80],
                    'max_features': [3],
                    'min_samples_leaf': [5],
                    'min_samples_split': [12],
                    'n_estimators': [100]
                }  

                # Initialize models
                rf_regressor = RandomForestRegressor()
                #dt_regressor = DecisionTreeRegressor()

                # Instantiate the grid search model
                grid_search = GridSearchCV(estimator = rf_regressor, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

                # Train models
                # Fit the grid search to the data
                grid_search.fit(X_train, y_train)

                best_params_path = os.path.join(self.config.root_dir_params, "best_params.pkl")
                os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
                joblib.dump(grid_search.best_params_, best_params_path)

                # Save trained model
                os.makedirs("model", exist_ok=True)
                model_path = os.path.join("model", "random_forest_model.pkl")
                joblib.dump(grid_search.best_estimator_, model_path)

                logger.info("Model training completed successfully.")
            else:
                logger.error("No CSV file found.")
                return None
            
        except Exception as e:
            logger.exception("Error occurred during model training:")
            raise e

