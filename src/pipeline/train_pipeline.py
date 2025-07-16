import sys
import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException



class TrainingPipeline:

    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            feature_store_file_path = data_ingestion.initiate_data_ingestion()
            return feature_store_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
        


    def start_data_transformation(self, feature_store_file_path):
        try:
            data_transformation = DataTransformation(feature_store_file_path)
            train_df, test_df, preprocessor_path = data_transformation.initiate_data_transformation()

            return train_df, test_df, preprocessor_path
        
        except Exception as e:
            raise CustomException(e, sys)
        

    
    def start_model_training(self, train_df, test_df):
        try:
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(
                train_df, test_df
            )
            return model_score
        
        except Exception as e:
            raise CustomException(e, sys)
        


    def run_pipeline(self):
        try:
            feature_store_file_path = self.start_data_ingestion()
            train_df, test_df, preprocessor_path = self.start_data_transformation(feature_store_file_path)
            roc_auc_score = self.start_model_training(train_df, test_df)

            print("Training completed. Trained model ROC-AUC score:", roc_auc_score)

        except Exception as e:
            raise CustomException(e, sys)