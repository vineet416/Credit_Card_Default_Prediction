import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    artifact_dir: str = os.path.join(artifact_folder)
    transformed_train_file_path: str = os.path.join(artifact_dir, 'train.npy')
    transformed_test_file_path: str = os.path.join(artifact_dir, 'test.npy')
    transformed_object_file_path: str = os.path.join(artifact_dir, 'preprocessor.pkl')


class DataTransformation:
    def __init__(self, feature_store_file_path):
        self.feature_store_file_path = feature_store_file_path

        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtils()


    @staticmethod
    def get_data(feature_store_file_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(feature_store_file_path)
            data.rename(columns={"default.payment.next.month": TARGET_COLUMN}, inplace=True)

            return data
        
        except Exception as e:
            raise CustomException(e, sys)
        

    
    def initiate_feature_engineering(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logging.info("Initiating feature engineering")

        try:
            dataframe['Avg_Bill_Amt'] = dataframe[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
                                                'BILL_AMT5', 'BILL_AMT6']].mean(axis=1).round(2)

            dataframe['Avg_Pay_Amt'] = dataframe[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4',
                                                'PAY_AMT5', 'PAY_AMT6']].mean(axis=1).round(2)

            dataframe['Avg_Delay_Score'] = dataframe[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1).round(2)

            dataframe['Avg_Credit_Utilization_Ratio'] = (dataframe['Avg_Pay_Amt'] / dataframe['LIMIT_BAL']).round(2)

            logging.info("Feature engineering completed successfully")
            return dataframe
        
        except Exception as e:
            raise CustomException(e, sys)



    def get_data_transformer_object(self):
        try:
            numerical_columns = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'Avg_Bill_Amt', 'Avg_Pay_Amt', 'Avg_Credit_Utilization_Ratio']

            nominal_columns = ['SEX', 'MARRIAGE']

            ordinal_columns = ['EDUCATION', 'Age_Groups', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'Avg_Delay_Score']

            numerical_pipeline = Pipeline(steps=[('imputation', SimpleImputer(strategy='median')),
                                                ('scaling', StandardScaler())])

            nominal_pipeline = Pipeline(steps=[('imputation', SimpleImputer(strategy='most_frequent')),
                                                ('encoding', OneHotEncoder(handle_unknown='ignore', drop='first'))])
                
            ordinal_pipeline = Pipeline(steps=[('imputation', SimpleImputer(strategy='most_frequent')),
                                            ('encoding', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

            preprocessor = ColumnTransformer([("numerical_pipeline", numerical_pipeline, numerical_columns),
                                                    ("nominal_pipeline", nominal_pipeline, nominal_columns),
                                                    ("ordinal_pipeline", ordinal_pipeline, ordinal_columns)])
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        


    def initiate_data_transformation(self):
        logging.info("Entered initiate_data_transformation method of DataTransformation class")

        try:
            dataframe = self.get_data(feature_store_file_path = self.feature_store_file_path)

            dataframe['Age_Groups'] = pd.cut(dataframe['AGE'], bins=[20, 25, 30, 35, 40, 45, 50, 55, 60, np.inf], 
                                             labels=['20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60+'], 
                                             right=False)
            
            dataframe.drop(columns=['AGE', 'ID'], inplace=True)
            dataframe = dataframe.drop_duplicates()

            dataframe = dataframe.replace({'MARRIAGE': {0: np.nan}})
            dataframe = dataframe.replace({'EDUCATION': {4: 0, 5: 0, 6: 0}})

            dataframe = self.initiate_feature_engineering(dataframe)

            X = dataframe.drop(columns= TARGET_COLUMN)
            y = dataframe[TARGET_COLUMN]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

            preprocessor = self.get_data_transformer_object()

            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            preprocessor_path = self.data_transformation_config.transformed_object_file_path

            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

            self.utils.save_object(file_path= preprocessor_path, obj= preprocessor)

            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            return (train_arr, test_arr, preprocessor_path)
        
        except Exception as e:
            raise CustomException(e,sys)