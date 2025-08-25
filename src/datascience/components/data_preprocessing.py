from src.datascience import logger
from src.datascience.entity.config_entity import DataPreprocessingConfig
from src.datascience.components.configurations import ConfigurationManager
from pathlib import Path
import pandas as pd
import re
from src.datascience.constants import *
from src.datascience.utils.common import (read_yaml, create_directories)
from src.datascience import logger

class DataPreprocessing:
    def __init__(self):
        pass

    def _init(self,data_preprocessing_config:DataPreprocessingConfig):
        self.data_preprocessing_config = data_preprocessing_config

    def _load(self):
        df = pd.read_csv(self.data_preprocessing_config.source_file)
        return df

    def _drop_columns(self, df:pd.DataFrame):
        for column in self.data_preprocessing_config.column_names:
            df = df.drop(columns=f'{column}')
        return df
    
    def _convert_usd_to_sar(self, df:pd.DataFrame):
        mask = df['Currency Code'] == 'USD'
        df.loc[mask, 'Total Bcy'] = df.loc[mask, 'Total Bcy'] * 3.75
        df.loc[mask, 'Currency Code'] = 'SAR'
        return df

    def _fill_missing_item_names(self, df:pd.DataFrame):
        df['Item Name'].fillna("Unknown Product", inplace=True)
        return df
    
    def _drop_zero_values_rows(self, df:pd.DataFrame):
        df = df[~((df['Total Bcy'] == 0) & (df['Quantity'] > 0))]
        return df
    
    def _feature_engineering(self, df:pd.DataFrame):
        df['Unit Price'] = df['Total Bcy'] / df['Quantity']
        return df
    
    def _normalise_whitespace_and_case(self, df: pd.DataFrame):
        x = df["Item Name"].astype(str)
        x = x.str.strip()
        x = x.str.replace(r"\s+", " ", regex=True)
        x = x.str.lower()
        df["Item Name"] = x
        return df
    
    def _normalise_arabic(self, df: pd.DataFrame):
        x = df["Item Name"].astype(str)
        tashkeel = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
        x = x.apply(lambda s: tashkeel.sub("", s))
        x = x.str.replace("ـ", "", regex=False)
        x = x.str.replace("[\u0623\u0625\u0622\u0671]", "ا", regex=True)
        x = x.str.replace("\u0649", "\u064A", regex=True)
        df["Item Name"] = x
        return df

    def _normalise_english(self, df: pd.DataFrame):
        x = df["Item Name"].astype(str)
        x = x.str.replace(r"[;,]", " ", regex=True)
        x = x.str.replace(r"[^\w\s\-/x\u0600-\u06FF]", " ", regex=True)
        x = x.str.replace(r"\s+", " ", regex=True).str.strip()
        df["Item Name"] = x
        return df
    
    def initiate_data_preprocessing(self):
        try:
            logger.info("\n<<<<< Data Preprocessing Initiated >>>>>\n")
            config = ConfigurationManager()
            data_preprocessing_config = config.get_data_preprocessing_config()
            data_preprocessing = DataPreprocessing()
            data_preprocessing._init(data_preprocessing_config)
            df = data_preprocessing._load()
            df = data_preprocessing._drop_columns(df)
            df = data_preprocessing._convert_usd_to_sar(df)
            df = data_preprocessing._fill_missing_item_names(df)
            df = data_preprocessing._drop_zero_values_rows(df)
            df = data_preprocessing._feature_engineering(df)
            df = data_preprocessing._normalise_whitespace_and_case(df)
            df = data_preprocessing._normalise_arabic(df)
            df = data_preprocessing._normalise_english(df)
            df.to_csv(Path(data_preprocessing_config.root_dir, data_preprocessing_config.csv_name), index=False)
            logger.info(f"\n<<<<< Data Preprocessing Completed CSV Saved at {Path(data_preprocessing_config.root_dir, data_preprocessing_config.csv_name)}>>>>>\n")

        except Exception as e:
            logger.exception(e)
            raise e
