import os
from pathlib import Path
from carPricePredictor.constants import *
from carPricePredictor.utils.common import read_yaml, create_directories
from carPricePredictor.entity.config_entity import (DataIngestionConfig, DataPreprocessConfig,
                                                TrainingConfig,
                                                EvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_preprocess_config(self) -> DataPreprocessConfig:
        config = self.config.data_preprocess

        create_directories([config.root_dir])

        data_preprocess_config = DataPreprocessConfig(
            root_dir=Path(config.root_dir),
            local_data_file=Path(config.local_data_file),            
            param_filter_make=self.params.get('NUMBER_OF_TOP_MAKE', 10),
            param_filter_model=self.params.get('NUMBER_OF_TOP_MODEL', 3)            
        )

        return data_preprocess_config


    def get_training_config(self) -> TrainingConfig:
        config = self.config.training

        create_directories([config.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            root_dir_data=Path(config.root_dir_data),
            trained_model_path=Path(config.trained_model_path),
            root_dir_X_test=Path(config.root_dir_X_test),
            root_dir_y_test=Path(config.root_dir_y_test),
            root_dir_params=Path(config.root_dir_params)

        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation
        create_directories([config.root_dir])

        eval_config = EvaluationConfig(
            root_dir=Path(config.root_dir),
            source_dir=Path(config.source_dir),
            dt_model_path=Path(config.dt_model_path),
            rf_model_path=Path(config.rf_model_path),
            mlflow_uri=str(config.mlflow_uri),
            source_dir_x_test=Path(config.source_dir_x_test),
            source_dir_y_test=Path(config.source_dir_y_test),
            source_dir_params=Path(config.source_dir_params)
        )
        return eval_config
