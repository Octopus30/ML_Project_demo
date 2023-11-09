import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join("artifacts","model.pkl")
    
class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('split train and test input data')
            X_train,Y_train,X_test,Y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'RandomForest':RandomForestRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'Gradient Boosting':GradientBoostingRegressor(),
                'Liner regression':LinearRegression(),
                'K-Neighbours classifier': KNeighborsRegressor(),
                'XGBClassifier':XGBRegressor(),
                'Cat boosting classifier': CatBoostRegressor(verbose =False),
                'Adaboost Classsifier':AdaBoostRegressor()
            }
            
            model_report:dict = evaluate_models(X_train=X_train,Y_train=Y_train,X_test = X_test,
                                               Y_test =Y_test,models =models)
            
            ################################################################
            
            best_model_score = max(sorted(model_report.values()))
            
            ################################
            
            #to get model from dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException('No best model found')
            else:
                logging.info(f"best model found on both training and testing dataset")
                
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )    
            
            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(Y_test,predicted)
            
            return r2_square          
            
            
            
        except Exception as e:
            raise CustomException(e,sys)