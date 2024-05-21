import os
import pandas as pd
from carPricePredictor import logger
from carPricePredictor.utils.common import save_json
from carPricePredictor.entity.config_entity import DataPreprocessConfig
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

class DataPreprocess:
    def __init__(self, config: DataPreprocessConfig):
        self.config = config
        self.df = None

    def load_data(self):
        try:
            data_files = os.listdir(self.config.local_data_file)
            if data_files:
                csv_file = os.path.join(self.config.local_data_file, data_files[0])
                self.df = pd.read_csv(csv_file)
                logger.info("Data was successfully read by pandas")
            else:
                logger.error("No CSV file found.")
                return None
        except Exception as e:
            logger.exception("Error loading data:")
            raise e

    def drop_missing_values(self):
        try:
            self.df.dropna(inplace=True)
        except Exception as e:
            logger.exception("Error dropping missing values:")
            raise e

    def filter_by_make(self):
        try:
            top_makers = self.df['make'].value_counts().nlargest(self.config.param_filter_make).index.tolist()
            self.df = self.df[self.df['make'].isin(top_makers)]
        except Exception as e:
            logger.exception("Error filtering by make:")
            raise e

    def remove_outliers(self):
        try:
            def remove_outliers_inner(model_data):
                lr = LinearRegression()
                X = model_data[['mmr']]
                y = model_data['sellingprice']
                lr.fit(X, y)
                residuals = y - lr.predict(X)
                threshold = 2 * residuals.std()
                return model_data[abs(residuals) <= threshold]

            unique_models = self.df['model'].unique()
            updated_data = [remove_outliers_inner(self.df[self.df['model'] == model]) for model in unique_models]
            self.df = pd.concat(updated_data, ignore_index=True)
        except Exception as e:
            logger.exception("Error removing outliers:")
            raise e

    def filter_by_model(self):
        try:
            dfs = []
            grouped = self.df.groupby('make')
            for make, group_df in grouped:
                sorted_group = group_df['model'].value_counts().sort_values(ascending=False)
                top_models = sorted_group.head(self.config.param_filter_model).index.tolist()
                filtered_group = group_df[group_df['model'].isin(top_models)]
                dfs.append(filtered_group)
            self.df = pd.concat(dfs)
        except Exception as e:
            logger.exception("Error filtering by model:")
            raise e

    def add_age_feature(self):
        try:
            self.df['saledate_clean'] = pd.to_datetime(self.df['saledate'].str.split(' ').str[:5].str.join(' '), format='%a %b %d %Y %H:%M:%S')
            self.df['age'] = self.df['saledate_clean'].dt.year - self.df['year']
            self.df.drop(columns=['saledate_clean'], inplace=True)
        except Exception as e:
            logger.exception("Error adding age feature:")
            raise e

    def drop_unnecessary_columns(self):
        try:
            self.df = self.df.drop(columns=['body', 'trim', 'vin', 'seller', 'mmr', 'saledate', 'state', 'interior', 'color'])
        except Exception as e:
            logger.exception("Error dropping unnecessary columns:")
            raise e

    def transform_to_numerical(self):
        try:
            df = df.apply(lambda x: x.str.upper() if x.dtype == "object" else x)  
            df = df.dropna(how='any', axis=0)
            le_make = LabelEncoder()
            df["make"] = le_make.fit_transform(df["make"])
            le_model = LabelEncoder()
            df["model"] = le_model.fit_transform(df["model"])
            le_transmission = LabelEncoder()
            df["transmission"] = le_transmission.fit_transform(df["transmission"])        
            root_path_encoder = self.config.root_dir_encoder
            os.makedirs(root_path_encoder, exist_ok=True)
            encoder_data = {"make": le_make, "model": le_model, "transmission": le_transmission}
            encoder_file_path = os.path.join(root_path_encoder, "saved_encoder_data.pkl")
            joblib.dump(encoder_data, encoder_file_path)
        except Exception as e:
            logger.exception("Error transforming to numerical:")
            raise e

    def store_preprocessed_data(self):
        try:
            root_path = self.config.root_dir
            os.makedirs(root_path, exist_ok=True)
            output_file_path = os.path.join(root_path, "preprocessed_data.csv")
            self.df.to_csv(output_file_path, index=False)
            logger.info("Preprocess steps were successfully done")
        except Exception as e:
            logger.exception("Error storing preprocessed data:")
            raise e


