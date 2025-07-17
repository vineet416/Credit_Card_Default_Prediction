import shutil
import os
import sys
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from src.logger import logging
from src.exception import CustomException
from src.constant import *
from src.utils.main_utils import MainUtils
from src.components. data_transformation import DataTransformation
from dataclasses import dataclass


@dataclass
class PredictionPipelineConfig:
    model_file_path: str = os.path.join(artifact_folder, 'model.pkl')
    preprocessor_path: str = os.path.join(artifact_folder, 'preprocessor.pkl')



class PredictionPipeline:
    def __init__(self, features: pd.DataFrame):
        self.features = features
        self.utils = MainUtils()
        self.data_transformation = DataTransformation(features)
        self.prediction_pipeline_config = PredictionPipelineConfig()


    def data_preprocessing(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            # Correcting missing values
            features = features.replace({'MARRIAGE': {0: np.nan}})
            features = features.replace({'EDUCATION': {4: 0, 5: 0, 6: 0}})

            # Feature engineering
            logging.info("Initiating feature engineering.")
            feature_engineered_data = self.data_transformation.initiate_feature_engineering(features)

            logging.info("Initiating data transformation.")
            # Data transformation
            preprocessor = self.utils.load_object(self.prediction_pipeline_config.preprocessor_path)
            transformed_data = preprocessor.transform(feature_engineered_data)

            logging.info("Data preprocessing completed successfully.")

            return transformed_data, preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        

        

    def predict_proba(self, transformed_x: np.ndarray) -> np.ndarray:
        try:
            model = self.utils.load_object(self.prediction_pipeline_config.model_file_path)
            probability = model.predict_proba(transformed_x)
            return probability, model
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def feature_importance(self, model, preprocessor) -> list:
        try:
            logging.info("Calculating feature importance.")
            feature_importances = model.feature_importances_
            feature_names = preprocessor.get_feature_names_out()
            top_features_indices = np.argsort(feature_importances)[-5:][::-1]
            top_features = feature_names[top_features_indices]
            top_features = [feature.split('__')[-1] for feature in top_features]

            # Converting to DataFrame with importances value
            top_features_df = pd.DataFrame({
                'Feature': top_features,
                'Importance': feature_importances[top_features_indices].round(4)
            })

            logging.info("Feature importance calculated successfully.")
            return top_features_df

        except Exception as e:
            raise CustomException(e, sys)

        
    def run_pipeline(self):
        try:
            logging.info("Starting prediction pipeline.")
            transformed_x, preprocessor = self.data_preprocessing(self.features)
            probability, model = self.predict_proba(transformed_x)
            top_features = self.feature_importance(model, preprocessor)

            logging.info("Prediction pipeline completed successfully.")
            return probability, top_features
        
        except Exception as e:
            raise CustomException(e, sys)