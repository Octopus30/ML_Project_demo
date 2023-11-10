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
                'Linear regression':LinearRegression(),
                'K-Neighbours classifier': KNeighborsRegressor(),
                'XGBClassifier':XGBRegressor(),
                'Cat boosting classifier': CatBoostRegressor(verbose =False),
                'Adaboost Classsifier':AdaBoostRegressor()
            }
            
            params = {
                "Decision Tree" : {
                    'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'splitter' : ['best','random'],
                    'max_depth' : [5,10,15,20],
                    #'max_features' : ['auto','sqrt','log2'],
                    
                },
                "RandomForest" : {
                    'n_estimators' : [8,16,32,64,128,256],
                    'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                'Gradient Boosting' : {
                    'loss' : ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'n_estimators' : [8,16,32,64,128,256],
                    'learning_rate' : [0.0001,0.001,0.01],
                    'subsample' : [0.6,0.7,0.75,0.8,0.85,0.9],
                    
                },
                'Linear regression' : {},
                
                "K-Neighbours classifier" : {
                    'algorithm' : ['auto','ball_tree','kd_tree','brute'],
                    'n_neighbors' : [5,7,9,11]
                },
                'XGBClassifier' : {
                    'learning_rate' : [0.0001,0.001,0.01],
                    'n_estimators' : [8,16,32,64,128,256],                    

                },
                'Cat boosting classifier' : {
                    'depth': [6,8,10],
                    'learning_rate' : [0.0001,0.001,0.01],
                    'iterations' : [30,50,100]

                },
                'Adaboost Classsifier' : {
                    'learning_rate' : [0.0001,0.001,0.01],
                    'n_estimators' : [8,16,32,64,128,256]

                }
                
                
            }
            
            model_report:dict = evaluate_models(X_train=X_train,Y_train=Y_train,X_test = X_test,
                                               Y_test =Y_test,models =models , param = params)
            
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