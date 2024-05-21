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

    def preprecoss_data(self):
        '''
        Fetch data from artifactory
        '''
        try:
            data_files = os.listdir(self.config.local_data_file)
            if data_files:
            # step 0: load the dataset
                csv_file = os.path.join(self.config.local_data_file, data_files[0])
                df = pd.read_csv(csv_file)
                logger.info("data was successfully read by pandas")

            # Step 1: Drop any rows with missing values
                df.dropna(inplace=True)

            # Step 2: Filter based on make
                top_10_makers = df['make'].value_counts().nlargest(self.config.param_filter_make).index.tolist()                
                df = df[df['make'].isin(top_10_makers)]

            # step 3: Remove outliers based on redisuals using a linear regression model with relation betweem mmr and sellingprice
                def remove_outliers(model_data):
                    # Fit a linear regression model
                    lr = LinearRegression()
                    X = model_data[['mmr']]
                    y = model_data['sellingprice']
                    lr.fit(X, y)
                    
                    # Calculate residuals
                    residuals = y - lr.predict(X)
                    
                    # Define a threshold (e.g., 2 standard deviations)
                    threshold = 2 * residuals.std()  # Adjust this threshold as needed
                    
                    # Remove outliers
                    model_data = model_data[abs(residuals) <= threshold]
    
                    return model_data
                
                unique_models = df['model'].unique()  # Changed 'df' to 'df_filtered'
                updated_data = []

                for model in unique_models:
                    model_data = df[df['model'] == model]  
                    
                    # Remove outliers based on residuals
                    model_data = remove_outliers(model_data)
                    updated_data.append(model_data)
                    
                    # Plot the graph
                    # sns.relplot(x='mmr', y='sellingprice', data=model_data, kind='scatter')  
                    # plt.title(f"Selling Price vs. MMR for {model}")
                    # plt.xlabel('MMR')
                    # plt.ylabel('Selling Price')
                    # plt.show()

                # Concatenate updated dataframes outside the loop
                df = pd.concat(updated_data, ignore_index=True) 


            #Step 4: filter based on model
                dfs = []

                # Group the DataFrame by 'make'
                grouped = df.groupby('make')

                for make, group_df in grouped:
                    sorted_group = group_df['model'].value_counts().sort_values(ascending=False)                    
                    top3_models = sorted_group.head(self.config.param_filter_model).index.tolist()                
                    filtered_group = group_df[group_df['model'].isin(top3_models)]          
                    dfs.append(filtered_group)

                # Concatenate all filtered DataFrames into one DataFrame
                df = pd.concat(dfs)

            #Step 5: Adding additional feature age
                df['saledate_clean'] = df['saledate'].str.split(' ').str[:5].str.join(' ')
                df['saledate_clean'] = pd.to_datetime(df['saledate_clean'], format='%a %b %d %Y %H:%M:%S')

                df['saledate_year'] = df['saledate_clean'].dt.year   
                df['age'] = df['saledate_year'] - df['year']

                df.drop(columns=['saledate_clean', 'saledate_year'], inplace=True)
            
            #Step 6: remove unnecessary columns
                df = df.drop(columns=['body', 'trim', 'vin', 'seller', 'mmr', 'saledate', 'state','interior','color'])

            #Step 7: Transorm object type to numerical values
                df = df.apply(lambda x: x.str.upper() if x.dtype == "object" else x)  
                df = df.dropna(how='any', axis=0)
                le_make = LabelEncoder()
                df["make"] = le_make.fit_transform(df["make"])
                le_model = LabelEncoder()
                df["model"] = le_model.fit_transform(df["model"])
                le_transmission = LabelEncoder()
                df["transmission"] = le_transmission.fit_transform(df["transmission"])
            #Step 8: Store Label data as artifacts
                os.makedirs("model", exist_ok=True)
                encoder_data = {"make": le_make, "model": le_model, "transmission": le_transmission}
                encoder_file_path = os.path.join("model", "saved_encoder_data.pkl")
                joblib.dump(encoder_data, encoder_file_path)


            #Step 9: output preprocessed_data.csv
                root_path = self.config.root_dir
                os.makedirs(root_path, exist_ok=True)
                
                output_file_path = os.path.join(root_path, "preprocessed_data.csv")
                df.to_csv(output_file_path, index=False)

                logger.info("preprocess steps were successfully done")

            else:
                logger.error("No CSV file found.")
                return None
        except Exception as e:
            logger.exception("Error loading data:")
            raise e
        


