import sys
from typing import Generator, List, Tuple
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import warnings
warnings.filterwarnings("ignore")

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass



@dataclass
class ModelTrainerConfig:
    artifact_folder= os.path.join(artifact_folder)
    trained_model_path= os.path.join(artifact_folder,"model.pkl" )
    expected_roc_auc_score= 0.7
    model_config_file_path= os.path.join('config','model.yaml')




class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()
        self.models = {
                        'XGBClassifier': XGBClassifier(),
                        'GradientBoostingClassifier' : GradientBoostingClassifier(),
                        'KNNClassifier' : KNeighborsClassifier(),
                        'RandomForestClassifier': RandomForestClassifier()
                        }



    def model_train_eval_with_tuning(self, X_train, y_train, X_test, y_test, models):
        try:
            logging.info("Entered model_train_eval_with_tuning method of ModelTrainer class")

            # Get all model parameter grids from model.yaml
            model_config = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)

            model_param_grids = {
                model_name: model_config["model_selection"]["model"][model_name]["search_param_grid"]
                for model_name in model_config["model_selection"]["model"]
            }
            
            report = {}
            best_models = {}

            for model_name in models.keys():     
                logging.info(f"Training and evaluating model: {model_name}")

                # Get the base model and parameter grid
                base_model = models[model_name]
                param_grid = model_param_grids[model_name]
                
                # Perform GridSearchCV
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit the grid search
                grid_search.fit(X_train, y_train)
                
                # Get the best model
                best_model = grid_search.best_estimator_
                best_models[model_name] = best_model
                
                # Make predictions on test set
                y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
                
                # Calculate test ROC AUC score
                model_roc_auc_score = float(roc_auc_score(y_test, y_test_pred_proba))
                
                # Store all metrics and best parameters
                report[model_name] = {
                    'Best_Params': grid_search.best_params_,
                    'CV_Score': round(float(grid_search.best_score_), 4),
                    'Test_ROC_AUC': round(model_roc_auc_score, 4)
                }

                # Log all models metrics
                logging.info(f"Best Parameters for {model_name}: {grid_search.best_params_}")
                logging.info(f"Cross-Validation Score: {grid_search.best_score_:.4f}")
                logging.info(f"Test ROC AUC Score: {model_roc_auc_score:.4f}")

                
            return report

        except Exception as e:
                raise CustomException(e, sys)




    def get_best_model(self, report):
            
        try:
            logging.info("Entered get_best_model method of ModelTrainer class")
            logging.info("Evaluating the best model based on ROC-AUC score")

            # Get the best model based on ROC-AUC score
            best_model_name = max(report.keys(), key=lambda x: report[x]['Test_ROC_AUC'])
            best_model_score = report[best_model_name]['Test_ROC_AUC']

            # getting best model parameters
            best_model_params = report[best_model_name]['Best_Params']      

            # Retrieve the best model object
            best_model_object = self.models[best_model_name]

            # Best model object with the best parameters
            best_model_object.set_params(**best_model_params)

            return best_model_name, best_model_object, best_model_score
        
        except Exception as e:
            raise CustomException(e,sys)





    def initiate_model_trainer(self, train_df, test_df):
        try:
            logging.info(f"Splitting training and testing input and target feature")

            X_train, y_train, X_test, y_test = (
                train_df.drop(columns=[TARGET_COLUMN]).values,
                train_df[TARGET_COLUMN].values,
                test_df.drop(columns=[TARGET_COLUMN]).values,
                test_df[TARGET_COLUMN].values
            )

            logging.info(f"Extracting model config file path")
            
            # Train and evaluate models with hyperparameter tuning
            model_report: dict = self.model_train_eval_with_tuning(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models=self.models)

            # Get the best model name, object, and score
            best_model_name, best_model, best_model_score = self.get_best_model(model_report)

            # Fit the best model on the entire training data
            best_model.fit(X_train, y_train)
            best_model_params = best_model.get_params()

            # Make predictions on the test set
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]

            # Calculate metrics for the best model
            best_model_roc_auc_score = roc_auc_score(y_test, y_pred_proba)
            best_model_f1_score = f1_score(y_test, y_pred)
            best_model_precision = precision_score(y_test, y_pred)
            best_model_recall = recall_score(y_test, y_pred)

           
            # printing the best model details
            print(f"Best Model: {best_model_name} with ROC-AUC Score: {best_model_roc_auc_score}")
            print(f"Best Model F1 Score: {best_model_f1_score:.4f}")
            print(f"Best Model Precision: {best_model_precision:.4f}")
            print(f"Best Model Recall: {best_model_recall:.4f}")
            print(f"Best Model Parameters: {best_model_params}")


            if best_model_roc_auc_score < self.model_trainer_config.expected_roc_auc_score:
                raise Exception(f"No best model found with an ROC-AUC score greater than the threshold {self.model_trainer_config.expected_roc_auc_score}")

            logging.info(f"Best found model on both training and testing dataset")
            logging.info(f"Best model name: {best_model_name} and ROC-AUC score: {best_model_roc_auc_score:.4f}")
            logging.info(f"Best model F1 score: {best_model_f1_score:.4f}")
            logging.info(f"Best model precision: {best_model_precision:.4f}")
            logging.info(f"Best model recall: {best_model_recall:.4f}")
            logging.info(f"Best model parameters: {best_model_params}")

            logging.info(f"Saving model at path: {self.model_trainer_config.trained_model_path}")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
           
            return best_model_roc_auc_score

        except Exception as e:
            raise CustomException(e, sys)