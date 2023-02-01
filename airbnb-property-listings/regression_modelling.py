import itertools
import joblib
import json
import numpy as np
import os
import pandas as pd
import tabular_data 
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(2)

clean_data = pd.read_csv("/Users/sarahaisagbon/Downloads/clean_tabular_data.csv")
#clean_data = pd.read_csv("tabular_data/clean_tabular_data.csv")
X, y = tabular_data.load_airbnb(clean_data, "Price_Night")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3)
    
def simple_regression_model(X_train, y_train, X_test):
    #Train a simple regression model
    model = SGDRegressor().fit(X_train, y_train)
    y_hat = model.predict(X_test)
    return y_hat

def evaluate_performance(X_train, y_train, X_test, y_test):
    y_hat = simple_regression_model(X_train, y_train, X_test)
    #evaluate the performance of thi simple regression model
    MSE =  mean_squared_error(y_test, y_hat)
    RMSE = mean_squared_error(y_test, y_hat, squared = False)
    R2 = r2_score(y_test, y_hat)
    linear_regression_metrics = MSE, RMSE, R2
    return linear_regression_metrics

def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_validation, y_validation, X_test, y_test, hyperpara_searchspace):
    fitted_list = []
    #get hyperparameters names
    keys = hyperpara_searchspace.keys()
    #get the list of values to try for each hyperparameter
    vals = hyperpara_searchspace.values()
    #iterate over all combinations of each list of values 
    for combo in itertools.product(vals):
        hyper_values_dict = dict(zip(keys, combo))
        model_list =   {
                    "SGDRegressor" : SGDRegressor,
                    "DecisionTreeRegressor" : DecisionTreeRegressor,                   
                    "RandomForestRegressor" : RandomForestRegressor,
                    "GradientBoostingRegressor" : GradientBoostingRegressor
        }
        #select one of the models in model_list and use the hyper_values_dict to create a model
        model = model_list[model_class](hyper_values_dict)
        #fit the model to the training set
        model.fit(X_train, y_train)
        #make prediction using validation features set
        y_hat_validation = model.predict(X_validation)
        #check the MSE, RMSE and R2
        validation_MSE =  mean_squared_error(y_validation, y_hat_validation)
        validation_RMSE = mean_squared_error(y_validation, y_hat_validation, squared = False)
        validation_R2 = r2_score(y_validation, y_hat_validation)
        #put the errors in a dictionary
        performance_metrics_dict = {"validation_MSE" : validation_MSE, "validation_RMSE": validation_RMSE, "validation_R2": validation_R2}
        #put the model details into a list
        model_details = [model, hyper_values_dict, performance_metrics_dict]
        #add details for this particular model to the fitted_list
        fitted_list.append(model_details)
    #find the model with the lowest RMSE and that will be chosen as the best model
    best_model_details = min(fitted_list, key=lambda x: x[2]["validation_RMSE"])
    return best_model_details

def tune_regression_model_hyperparameters(model_class, X_train, y_train, X_validation, y_validation, X_test, y_test, hyperpara_searchspace):
    models_list =   {
                    "SGDRegressor" : SGDRegressor,
                    "DecisionTreeRegressor" : DecisionTreeRegressor,                    
                    "RandomForestRegressor" : RandomForestRegressor,
                    "GradientBoostingRegressor" : GradientBoostingRegressor
                    }
    model = models_list[model_class]()
    
    GS = GridSearchCV(estimator=model,
                      param_grid=hyperpara_searchspace,
                      scoring= ["neg_mean_squared_error", "neg_root_mean_squared_error", "r2"],
                      refit= "neg_root_mean_squared_error"
                      )
    
    GS.fit(X_train, y_train)
    best_model = models_list[model_class](**GS.best_params_)
    fitted_best_model = best_model.fit(X_train, y_train)
    #make prediction using validation features set
    y_hat_validation = best_model.predict(X_validation)
    #check the MSE, RMSE and R2
    validation_MSE =  mean_squared_error(y_validation, y_hat_validation)
    validation_RMSE = mean_squared_error(y_validation, y_hat_validation, squared = False)
    validation_R2 = r2_score(y_validation, y_hat_validation)
    #put the errors in a dictionary
    performance_metrics_dict = {"validation_MSE" : validation_MSE, "validation_RMSE": validation_RMSE, "validation_R2": validation_R2}
    #find the model with the lowest RMSE and that will be chosen as the best model
    best_model_details = [fitted_best_model, GS.best_params_, performance_metrics_dict]
    return best_model_details


def save_model(model_details, folder="models/regression/linear_regression"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    #get the model from the list and save it as joblib file
    model = model_details[0]
    joblib.dump(model, f"{folder}/model.joblib")
    
    #get the hyperparameters from the list and save it as json file
    hyperparameters = model_details[1]
    with open(f"{folder}/hyperparameters.json", "w") as fp:
        json.dump(hyperparameters, fp)
        
    #get the performance from the list and save it as json file
    performance_metrics = model_details[2]
    with open(f"{folder}/metrics.json", "w") as fp:
        json.dump(performance_metrics, fp)
    
def evaluate_all_models(task_folder="models/regression"):
    np.random.seed(2)
    #specify the search spaces for hyperparameters in each model
    stochastic_gradient_descent_model = tune_regression_model_hyperparameters("SGDRegressor", X_train, y_train, X_validation, y_validation, X_test, y_test, hyperpara_searchspace = 
    {
    "penalty" : ["l2", "l1"],
    "max_iter" : [250, 500, 750, 1000]
    })
    
    print("Evaluation of Stochastic Gradient Descent Model Complete!")
    #save the decision model in a folder called decision_tree
    save_model(stochastic_gradient_descent_model, folder=f"{task_folder}/SDG")
    print("Stochastic Gradient Descent Saved!")
    
    decision_tree_model = tune_regression_model_hyperparameters("DecisionTreeRegressor", X_train, y_train, X_validation, y_validation, X_test, y_test, hyperpara_searchspace = 
    {
    "criterion" : ["squared_error", "absolute_error"],
    "max_depth" : [20, 25, 30, 35, 40], #initially the set was [10, 20, 30, 40] then I got 30 as an answer so I narrowed down the set.
    "min_samples_split" : np.arange(0.2, 1, 0.2), #initial set = [0.2, 0.4, 2, 4] and got 0.4 
    "max_features" : [6, 7, 8, 9] #initial set = [2, 4, 6, 8] and got 8
    })
    
    print("Evaluation of Decision Tree Model Complete!")
    #save the decision model in a folder called decision_tree
    save_model(decision_tree_model, folder=f"{task_folder}/decision_tree")
    print("Decision Tree Model Saved!")
    
    random_forest_model = tune_regression_model_hyperparameters("RandomForestRegressor", X_train, y_train, X_validation, y_validation, X_test, y_test, hyperpara_searchspace =
    {
    "n_estimators" : [40, 50, 60], #initial set = [50, 100, 150, 200] and got 50
    "criterion" : ["squared_error", "absolute_error"],
    "max_features" : [0.1, 0.2, 0.3], #initially [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] and got 0.3
    "max_depth" : [25, 30, 40], # [3,5,7, 9] and got 9 then [9, 20, 30] and got 30
    "min_samples_split" : [4, 5, 6] #min_sample_split is either a float in (0, 1] or int initial set = [0.2, 0.4, 2, 4] and got 4
    })
    
    print("Evaluation of Random Forest Model Complete!")
    #save the Random Forest Model in a folder called random_forest
    save_model(random_forest_model, folder=f"{task_folder}/random_forest")
    print("Random Forest Model Saved!")
    
    gradient_boosting_model = tune_regression_model_hyperparameters("GradientBoostingRegressor", X_train, y_train, X_validation, y_validation, X_test, y_test, hyperpara_searchspace =
    {
    "learning_rate" : [0.05, 0.1, 0.15], #[0.1, 0.3] and got 0.1
    "loss" : ["squared_error", "absolute_error"],
    "n_estimators" : [50, 80, 100], #[20, 50, 100] and got 100
    "max_depth" : [3, 5, 7], 
    "max_features" : [2, 3, 4] #[2, 4, 6] and got 2
    })

    print("Evaluation of Gradient Boosting Model Complete!")
    #save the Gradient Boosting Model in a folder called gradient_boosting
    save_model(gradient_boosting_model, folder=f"{task_folder}/gradient_boosting")
    print("Gradient Boosting Model Saved!")

    return decision_tree_model, random_forest_model, gradient_boosting_model

def find_best_model(model_details_list, best_model_indicator, task_folder):
    #list of all the models in the task_folder
    list_of_files = os.listdir(task_folder)
    validation_scores = []  
    #list of all validation_scores in the model_details_list
    overall_score_list = [x[2][best_model_indicator] for x in model_details_list]
    #loop through the list of all models in the task_folder
    for file_name in list_of_files:
        #make the string lowercase and remove _
        file_name = file_name.lower().replace("_", "")
        #create a diction with the model names as keys and for each model, their validation_score dictionary as values
        model_names_key = [x[0] for x in model_details_list]
        model_names_value = [x[2] for x in model_details_list]
        model_names = dict(zip(model_names_key, model_names_value))
        #loop through the model names
        for model in model_names.keys():
            #compare the file names to the lowercase string of the model names 
            if file_name in str(model).lower():
                #if equal, get the particular_validation_score we want and put it into validation_scores
                particular_model_scores = model_names.get(model)
                validation_scores.append(particular_model_scores.get(best_model_indicator))
            else:
                pass
    
    #if these are regression models, we will be looking for the smallest error and there need to minimise the error.
    if "regression" in task_folder:
        best_model_index = overall_score_list.index(min(validation_scores))
    #However if these are classification models, we will be looking to maximise the score i.e. accuracy.
    elif "classification" in task_folder:
        best_model_index = overall_score_list.index(max(validation_scores))
    
    #In both cases, we find the index of the score in the overall_score_list and then we use that index to find all the details for that particular model.
    best_model_details = model_details_list[best_model_index]
    return best_model_details
    
if __name__ == "__main__":
    np.random.seed(2)
    task_folder = "models/regression"
    model_details_list = evaluate_all_models(task_folder)
    best_model = find_best_model(model_details_list, "validation_RMSE", task_folder)
    print(best_model)