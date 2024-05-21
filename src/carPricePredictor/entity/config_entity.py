from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataPreprocessConfig:
    root_dir: Path
    #root_dir_encoder: Path
    local_data_file: Path
    param_filter_make: int
    param_filter_model: int

# @dataclass(frozen=True)
# class PrepareBaseModelConfig:
#     root_dir: Path
#     base_model_path: Path
#     updated_base_model_path: Path
#     params_image_size: list
#     params_learning_rate: float
#     params_include_top: bool
#     params_weights: str
#     params_classes: int




@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    root_dir_data: Path
    trained_model_path: Path
    root_dir_X_test: Path
    root_dir_y_test: Path
    root_dir_params: Path
    # updated_base_model_path: Path
    # training_data: Path
    # params_epochs: int
    # params_batch_size: int
    # params_is_augmentation: bool
    # params_image_size: list



@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    source_dir: Path
    dt_model_path: Path
    rf_model_path: Path
    mlflow_uri: str
    source_dir_x_test: Path
    source_dir_y_test: Path
    source_dir_params: Path

