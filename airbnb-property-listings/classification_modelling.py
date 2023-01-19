import numpy as np
import os
import pandas as pd
import regression_modelling
import tabular_data 
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

np.random.seed(2)

clean_data = pd.read_csv("/Users/sarahaisagbon/Downloads/clean_tabular_data.csv")
#clean_data = pd.read_csv("tabular_data/clean_tabular_data.csv")
X, label = tabular_data.load_airbnb(clean_data, "Category")

#Encode the labels 
lb = LabelEncoder()
y = lb.fit_transform(label)

X, y = (X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)


def simple_classification_model(X_train, y_train, X_test):
    #Train a simple classification model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    return y_hat

def evaluate_performance(X_train, y_train, X_test, y_test):
    y_hat = simple_classification_model(X_train, y_train, X_test)
    #evaluate the performance of this simple classification model
    #Initially used micro averaging but I got the same value for each
    Accuracy =  accuracy_score(y_test, y_hat)
    Precision = precision_score(y_test, y_hat, average='macro', zero_division=0)
    Recall = recall_score(y_test, y_hat, average='macro')
    F1 = f1_score(y_test, y_hat, average='macro')
    linear_regression_metrics = {'Model_Accuracy': Accuracy, 'Model_F1_Score': F1, 'Model_Precision_Score': Precision, 'Model_Recall_Score': Recall }
    return linear_regression_metrics

def tune_classification_model_hyperparameters(model_class, X_train, y_train, X_validation, y_validation, X_test, y_test, hyperpara_searchspace):
    np.random.seed(2)
    models_list =   {
                    "LogisticRegression" : LogisticRegression,
                    "DecisionTreeClassifier" : DecisionTreeClassifier,                   
                    "RandomForestClassifier" : RandomForestClassifier,
                    "GradientBoostingClassifier" : GradientBoostingClassifier
                    }
    model = models_list[model_class]()
    
    GS = GridSearchCV(estimator=model,
                      param_grid=hyperpara_searchspace,
                      scoring= ["accuracy", "recall_macro", "f1_macro"],
                      refit= "accuracy"
                      )
    
    GS.fit(X_train, y_train)
    best_model = models_list[model_class](**GS.best_params_)
    fitted_best_model = best_model.fit(X_train, y_train)
    #make prediction using validation features set
    y_hat_validation = best_model.predict(X_validation)
    #check the accuracy, precision, recall and F1
    validation_accuracy =  accuracy_score(y_validation, y_hat_validation)
    #validation_precision = precision_score(y_validation, y_hat_validation, average="macro", zero_division=1)
    validation_recall = recall_score(y_validation, y_hat_validation, average="macro")
    validation_F1 = f1_score(y_validation, y_hat_validation, average="macro")
    #put the errors in a dictionary
    performance_metrics_dict = {"validation_accuracy" : validation_accuracy, "validation_recall": validation_recall, "validation_F1": validation_F1}
    #find the model with the highest accuracy and that will be chosen as the best model
    best_model_details = [fitted_best_model, GS.best_params_, performance_metrics_dict]
    return best_model_details

def save_model(model_details, folder="models/classification/logistic_regression"):
    regression_modelling.save_model(model_details, folder)
    
def evaluate_all_models(task_folder="models/classification"):
    np.random.seed(2)
    #specify the serach spaces for hyperparameters in each model
    logistic_regression_model = tune_classification_model_hyperparameters("LogisticRegression", X_train, y_train, X_validation, y_validation, X_test, y_test, hyperpara_searchspace = 
    {
    "max_iter" : [100, 500, 1000], 
    "solver" : ["newton-cg", "sag", "saga", "lbfgs"], #suggested hyperparameter with solvers that work for multiclass classification
    "multi_class": ["auto", "multinomial"] #another suggested hyperparameter with values that ork for multiclass classification
    })
    
    print("Evaluation of Logistic Regression Model Complete!")
    #save the decision model in a folder called logistic_regression
    save_model(logistic_regression_model, folder=f"{task_folder}/logistic_regression")
    print("Logistic Regression Saved!")
    
    decision_tree_model = tune_classification_model_hyperparameters("DecisionTreeClassifier", X_train, y_train, X_validation, y_validation, X_test, y_test, hyperpara_searchspace = 
    {

    "criterion" : [ "gini", "entropy", "log_loss"],
    "max_depth" : [5, 6, 7, 8, 9, 10], #[10, 20, 30, 40],
    "min_samples_split" : [2, 3, 4], #[0.2, 0.4, 2, 4], and got 2
    "max_features" : [6, 7, 8, 9, 10] #[2, 4, 6, 8] and got 8
    })
    
    print("Evaluation of Decision Tree Model Complete!")
    #save the decision model in a folder called decision_tree
    save_model(decision_tree_model, folder=f"{task_folder}/decision_tree")
    print("Decision Tree Model Saved!")
    
    np.random.seed(2)
    random_forest_model = tune_classification_model_hyperparameters("RandomForestClassifier", X_train, y_train, X_validation, y_validation, X_test, y_test, hyperpara_searchspace =
    {
    "n_estimators" : [60, 70, 80], #[50, 100] and got 50
    "criterion" : ["gini", "entropy", "log_loss"],
    "max_features" : [0.5, 0.6, 0.7], #[0.2, 0.4, 0.6] and got 0.6
    "max_depth" : [5, 6, 7], #[8, 9, 10], and got 8 [9, 20] and got 9
    "min_samples_split" : [2, 3, 4]
    })
    
    print("Evaluation of Random Forest Model Complete!")
    #save the Random Forest Model in a folder called random_forest
    save_model(random_forest_model, folder=f"{task_folder}/random_forest")
    print("Random Forest Model Saved!")
    
    np.random.seed(2)
    gradient_boosting_model = tune_classification_model_hyperparameters("GradientBoostingClassifier", X_train, y_train, X_validation, y_validation, X_test, y_test, hyperpara_searchspace =
    {
    "learning_rate" : [0.05, 0.1, 0.2],
    "loss" : ["log_loss"],
    "n_estimators" : [7, 8, 9], #[5, 8, 10] and got 8, #[10, 20, 30], and got 10 [30, 50, 80], and got 30
    "max_depth" : [3, 4, 5], #[5, 7, 8], and got 5
    "max_features" : [2, 3, 4] #[1, 4, 9, 16, 25] and got 4
    })

    print("Evaluation of Gradient Boosting Model Complete!")
    #save the Gradient Boosting Model in a folder called gradient_boosting
    save_model(gradient_boosting_model, folder=f"{task_folder}/gradient_boosting")
    print("Gradient Boosting Model Saved!")

    return decision_tree_model, random_forest_model, gradient_boosting_model

def find_best_model(model_details_list, best_model_indicator, task_folder):
    #adapted the find_best_model in regression_modelling.py to work for a specified best_model_indicator and task_folder.
    best_model_details = regression_modelling.find_best_model(model_details_list, best_model_indicator, task_folder)
    return best_model_details

if __name__ == "__main__":
    task_folder="models/classification"
    model_details_list = evaluate_all_models(task_folder)
    best_model = find_best_model(model_details_list, "validation_accuracy", task_folder)
    print(best_model)