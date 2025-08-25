from src.datascience.components.data_ingestion import DataIngestion
from src.datascience.components.data_preprocessing import DataPreprocessing
from src.datascience.components.data_categorisation import DataCategorisation
from src.datascience import logger

#After Running main.py Run (mlflow ui) in terminal
try:
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()

    data_preprocessing = DataPreprocessing()
    data_preprocessing.initiate_data_preprocessing()

    data_categorisation = DataCategorisation()
    data_categorisation.initiate_data_categorisation()
except Exception as e:
    logger.exception(e)
    raise e