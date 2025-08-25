from src.datascience import logger
from src.datascience.entity.config_entity import DataIngestionConfig
from src.datascience.components.configurations import ConfigurationManager
from pathlib import Path
import pandas as pd


class DataIngestion:
    def __init__(self):
        pass

    def _init(self, data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
    
    def _xlsx_to_csv(self):
        try:
            xlsx_data = pd.read_excel(Path(self.data_ingestion_config.source_file), sheet_name=0)
            xlsx_data.to_csv(Path(self.data_ingestion_config.root_dir, self.data_ingestion_config.csv_name), index=False)
        except Exception as e:
            logger.exception(e)
            raise e

    def initiate_data_ingestion(self):
        try:
            logger.info("<<<Initiating Data Ingestion>>>")
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion()
            data_ingestion._init(data_ingestion_config)
            data_ingestion._xlsx_to_csv()
            logger.info(f"csv file saved at {Path(data_ingestion_config.root_dir, data_ingestion_config.csv_name)}")
            logger.info("<<<<Data Ingestion Completed>>>>")


        except Exception as e:
            logger.exception(e)
            raise e