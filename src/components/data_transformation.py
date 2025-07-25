import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    artifact_dir: str = os.path.join(artifact_folder)
    transformed_train_file_path: str = os.path.join(artifact_dir, 'train.csv')
    transformed_test_file_path: str = os.path.join(artifact_dir, 'test.csv')
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

            # Renaming the target column
            data.rename(columns={"default.payment.next.month": TARGET_COLUMN}, inplace=True)

            # Sort the DataFrame by 'ID' column if it exists
            if 'ID' in data.columns:
                data = data.sort_values(by='ID').reset_index(drop=True)

            # Correcting data types
            for col in data.columns:
                if col not in ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']:
                    data[col] = data[col].astype('int64')

            return data
        
        except Exception as e:
            raise CustomException(e, sys)


    
    def initiate_feature_engineering(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logging.info("Initiating feature engineering")

        try:
            # Age Groups
            dataframe['Age_Groups'] = pd.cut(dataframe['AGE'], bins=[20, 25, 30, 35, 40, 45, 50, 55, 60, np.inf], 
                                        labels=['20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60+'], 
                                        right=False)
            dataframe = dataframe.drop(columns=['AGE'])

            # Average Bill Amount
            dataframe['Avg_Bill_Amt'] = dataframe[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1).round(2)

            # Average Payment Amount
            dataframe['Avg_Pay_Amt'] = dataframe[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].mean(axis=1).round(2)

            # Average Delay Score Calculation
            dataframe['Avg_Delay_Score'] = dataframe[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1).round(2)
            
            # Average Credit Utilization Ratio Calculation
            bill_amt_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
            utilization_ratios = []
            for col in bill_amt_cols:
                dataframe[f'UTIL_{col}'] = dataframe[col] / dataframe['LIMIT_BAL']
                utilization_ratios.append(f'UTIL_{col}')
            dataframe['Average_Credit_Utilization_Ratio'] = dataframe[utilization_ratios].mean(axis=1).round(2)
            dataframe = dataframe.drop(columns=[col for col in dataframe.columns if 'UTIL_' in str(col)])

            # droping 'ID' column if exists
            if 'ID' in dataframe.columns:
                dataframe = dataframe.drop(columns=['ID'])

            logging.info("Feature engineering completed successfully")
            return dataframe
        
        except Exception as e:
            raise CustomException(e, sys)



    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                'LIMIT_BAL',
                'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 
                'Avg_Bill_Amt', 'Avg_Pay_Amt', 'Avg_Delay_Score', 'Average_Credit_Utilization_Ratio'
            ]

            nominal_columns = ['SEX', 'MARRIAGE']

            ordinal_columns = ['EDUCATION', 'Age_Groups', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

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

            # correcting missing values
            dataframe = dataframe.replace({'MARRIAGE': {0: np.nan}})
            dataframe = dataframe.replace({'EDUCATION': {4: 0, 5: 0, 6: 0}})

            # Performing feature engineering
            dataframe = self.initiate_feature_engineering(dataframe)
            
            # Drop duplicates if any
            if dataframe.duplicated().any():
                dataframe = dataframe.drop_duplicates()

            # Splitting the data into features and target variable
            X = dataframe.drop(columns= TARGET_COLUMN)
            y = dataframe[TARGET_COLUMN]

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

            # Getting the data transformer object
            preprocessor = self.get_data_transformer_object()

            # Fitting and transforming the training data, and transforming the test data
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Applying SMOTE for balancing the training dataset
            smote = SMOTE(sampling_strategy=0.7, random_state=1)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

            preprocessor_path = self.data_transformation_config.transformed_object_file_path

            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

            self.utils.save_object(file_path= preprocessor_path, obj= preprocessor)

            # Constructing the final training and test arrays
            train_arr = np.c_[X_train_balanced, np.array(y_train_balanced)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            # Get feature names from the fitted preprocessor
            feature_names = preprocessor.get_feature_names_out()

            # Adding target column name
            all_column_names = list(feature_names) + [TARGET_COLUMN] 

            # removing prefix from all column names
            all_column_names = [name.split('__')[-1] for name in all_column_names]

            # Converting train_arr to DataFrame
            train_df = pd.DataFrame(train_arr, columns=all_column_names)

            # converting test_arr to DataFrame
            test_df = pd.DataFrame(test_arr, columns=all_column_names)

            # Saving the transformed train and test data to CSV files
            train_df.to_csv(self.data_transformation_config.transformed_train_file_path, index=False)
            test_df.to_csv(self.data_transformation_config.transformed_test_file_path, index=False)

            return (train_df, test_df, preprocessor_path)

        except Exception as e:
            raise CustomException(e,sys)