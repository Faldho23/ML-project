import os 

import numpy as np
import pandas as pd
import dill
from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e)

def evaluate_models(X_train, y_train, X_test, y_test, models, params, cv=3):
    try:
        report={}

        for model_name, model in models.items():
            gs = GridSearchCV(
                estimator=models[model_name],
                param_grid=params[model_name],
                cv=cv)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train_pred, y_train)
            test_model_score = r2_score(y_test_pred, y_test)

            report[model_name] = test_model_score
        
        return report
    
    except Exception as e:
        raise CustomException(e)
