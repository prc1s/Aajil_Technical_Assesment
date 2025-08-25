from src.datascience.constants import *
from src.datascience.utils.common import (read_yaml, create_directories)
from src.datascience.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig, DataCategorisationConfig

class ConfigurationManager:
    def __init__(self,
                 config_file_path=CONFIG_FILE_PATH):
        self.config=read_yaml(config_file_path)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config=self.config.data_ingestion.save_csv
        create_directories([config.root_dir])

        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_file=config.source_file,
            csv_name=config.csv_name
        )
        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        create_directories([config.root_dir])
        
        data_preprocessing_config = DataPreprocessingConfig(
            root_dir= config.root_dir,
            source_file= config.source_file,
            csv_name= config.csv_name,
            column_names = config.drop_columns.column_names
        )
        return data_preprocessing_config
    
    def get_data_categorisation_config(self) -> DataCategorisationConfig:
        config = self.config.data_categorisation
        model_config = config.model
        params_config = config.params
        mlflow_config = config.mlflow
        create_directories([config.root_dir])
        
        data_categorisation_config = DataCategorisationConfig(
            root_dir = config.root_dir,
            source = config.source,
            seeds = config.seeds,

            model_name = model_config.model_name,
            batch_size = model_config.batch_size,
            normalise = model_config.normalise,

            use_pca = params_config.use_pca,
            pca_n = params_config.pca_n,
            k_grid = params_config.k_grid,
            conf_threshold = params_config.conf_threshold,
            random_state = params_config.random_state,
            tau = params_config.tau,

            experiment = mlflow_config.experiment,
            run_name = mlflow_config.run_name
        )
        return data_categorisation_config