# **Modelling Airbnb's property listing dataset**

# Milestone 3

### **Data Preparation**
There were three main tasks in this milestone. Firstly, created a file called tabular_data where I loaded the tabular dataset and cleaned it. Once cleaned, I saved it as *clean_tabular_data.csv*. Next, I handled the image data in a file called prepare_image_data. I checked that all images were RGB and then resized them. Once the images have been resized, I created a new folder called *processed_images* and saved them in there. The final task was about preparing the tabular data for machine learning modelling. I did this in the *tabular_data* file. This involved getting the features and the label vector and putting it into a tuple.

### Code Used: *tabular_data.py*

```
import pandas as pd

def remove_rows_with_missing_ratings(df_raw):
    #Drop rows with missing values
    #cols = list(df_raw.filter(like='rating').columns)
    rating_cols = [col for col in df_raw.columns if "rating" in col]
    df=df_raw.dropna(subset=rating_cols)
    # Reset index after drop
    df=df_raw.dropna(subset=rating_cols).reset_index(drop=True)
    return df 

def convert_description(lst_of_str):
    #Remove [About this space and ] 
    lst_of_str = str(lst_of_str)[22: -1]
    #Turn the string into a list
    lst_of_str = lst_of_str.split(", '")
    #Create a list of strings we will remove from the list of strings
    remove_str = ["The space'", "", "Guest access'", "Other things to note'", "'"] 

    #remove ' from each string in the list
    for ele in lst_of_str:
        ele = ele[0:-1]

    #remove elements in the remove_str from the list of strings
    for string in remove_str:
        if string in lst_of_str:
            lst_of_str.remove(string)
        else:
            pass
    
    #Turn the list of strings into a string
    lst_of_str = ','.join(lst_of_str)
    
    #Clean up the string and remove excess commas, full stops, spaces and quotation marks
    lst_of_str = lst_of_str.replace("',", " ")
    lst_of_str = lst_of_str.replace('",', " ")
    lst_of_str = lst_of_str.replace(".'", ".")
    lst_of_str = lst_of_str.replace('.   ', ". ")
    return lst_of_str

def combine_description_strings(df_raw):
    #make a copy of the dataframe
    df=df_raw.copy()
    #apply the function convert_description to the Description column
    df['Description'] = df['Description'].apply(convert_description)
    return df

def set_default_feature_values(df_raw):
    #set the na values in the feature columns as 1
    df=df_raw.copy()
    columns = ["guests", "beds", "bathrooms", "bedrooms"]
    df[columns]=df[columns].fillna(1)
    return df
    
def clean_tabular_data(df_raw):
    #calls each function sequentially
    df_raw2 = remove_rows_with_missing_ratings(df_raw)
    df_raw3 = combine_description_strings(df_raw2)
    df = set_default_feature_values(df_raw3)
    return df

def load_airbnb(df_raw, label):
    #
    df_numerical = df_raw.select_dtypes(include=np.number)
    label_vector = df_numerical[label]
    features = df_numerical.drop(label, axis=1)
    tup = (features, label_vector)
    return tup

```

### Code Used: *clean_tabular.data.py*

```
import glob
import numpy as np
import os
from PIL import Image

class PrepareImages:
    def __init__(self):
        self.imagepaths = []
        self.imageheights = []
        self.RGBimages = []
        self.resized_images = []
        
    def createFolder(self, path):
        '''
        This function is used to create a folder.
        
        Args:
            path: the string representation of the path for the new folder.
        '''
        try:
            if not os.path.exists(path):
                os.mkdir(path)
        except OSError:
            print ("Error: Creating directory. " +  path)
            pass
        
    def load_and_check_images(self):
        '''
        This function loads the images, checks if the images are RGB images and finds the height for each image.
        
        '''
        filepath = "/Users/sarahaisagbon/Documents/GitHub/Data-Science/airbnb-property-listings/images"
        #loop through the image folder and get the image for each subfolder
        self.imagepaths = glob.glob(filepath + '/**/*.png', recursive=True)
        
        #load each image
        for image in self.imagepaths:
            img = Image.open(image)
            img_arr = np.asarray(img)
            #first check if image is RGB, if not dicard
            if len(img_arr.shape)!=3:
                pass
            else:
                #create list of RGB image filepaths
                self.RGBimages.append(image)

                #get the height of each image and put it in a list
                height = img.height
                self.imageheights.append(height)

        print(f"There are {len(self.RGBimages)} RGB images")
    
    def resize_images(self):
        '''
        This function finds the minimum height alongst the images and resized all the RGB images.
        
        '''
        #find minimum height of all the images
        min_height = min(self.imageheights)
        
        for image in self.RGBimages:
            img = Image.open(image)
            height = img.height
            width = img.width
            #find the appropriate width for the new height
            new_width  = int(min_height * width / height)
            #resize all images to the same height and width as the smallest image
            new_image = img.resize((new_width, min_height))
            self.resized_images.append(new_image)
        print("All images resized!")
            
    def save_resized_image(self):
        '''
        This function creates processed_image folder and saved the resized image in the new folder.
        
        '''
        #create processed_images folder
        new_folder = "/Users/sarahaisagbon/Documents/GitHub/Data-Science/airbnb-property-listings/processed_images"
        PrepareImages.createFolder(self, new_folder)
        
        for old_img_path, img in zip(self.RGBimages, self.resized_images):
            imagepath = str(old_img_path).split("/")
            #save new version in processed_images folder
            new_imagepath = os.path.join(new_folder, imagepath[-1])
            img.save(new_imagepath)
        print("All resized images saved!")

    def processing(self):
        PrepareImages.load_and_check_images(self)
        PrepareImages.resize_images(self)
        PrepareImages.save_resized_image(self)

```

# Milestone 4

### **Create a regression model**
There were seven tasks in this milestone. 
1. Create a simple regression to predict the nightly cost of each listing. 
    - Firstly, I created a modelling.py file, which I put all the code. 
    - I imported the load_airbnb from the tabular_data.py file. 
    - I used the SGDRegressor model class to get a linear regressiom model.

### Code Used: *modelling.py*

```
clean_data = pd.read_csv("tabular_data/clean_tabular_data.csv")
X, y = tabular_data.load_airbnb(clean_data, "Price_Night")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3)
    
def simple_regression_model(X_train, y_train, X_test):
    #Train a simple regression model
    model = SGDRegressor().fit(X_train, y_train)
    y_hat = model.predict(X_test)
```

2. Evaluate the regression model performance
    - I used sklearn to compute the MSE, RMSE and R2.
    - These metrics evaluate the performance of the the regression model.

### Code Used: *modelling.py*

```
def evaluate_performance(y_test, y_hat):
    #evaluate the performance of thi simple regression model
    MSE =  mean_squared_error(y_test, y_hat)
    RMSE = mean_squared_error(y_test, y_hat, squared = False)
    R2 = r2_score(y_test, y_hat)
    linear_regression_metrics = MSE, RMSE, R2
    return linear_regression_metrics
```
3. Implement a custom function to tune the hyperparemeters of the model.
    - I created a function called custom_tune_regression_model_hyperparameters that implmented the Grid Serach method on the hyperparameters of several regression models.
    - The function returned the best model, a dictionary of the best hyperparameters for the model and a dictionary of performance metrics, which included the same metrics as the previous function.

### Code Used: *modelling.py*

```
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
```
4. Tune the hyperparameters using sklearn
    - I used the sklearn module to implement a grid-search on hyperparameters.
    - The function returned the same list as in Task 3.
### Code Used: *modelling.py*

```
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
    best_model = models_list[model_class]()
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

```
5. Save the model 
    - Firstly, I created a folder called models. 
    - I then defined the save_model function, which saved the model, hyperparameter dictionary and performance metric dictionary in different files in a folder with a folder path entered as an argument.

### Code Used: *modelling.py*

```
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
```
6. Beat the baseline regression model
    - I defined a function called evaluate_all_models.
    - I used the tune_regression_model_hyperparameters function to improve the hyperparameters of the Decision Tree, Random Forest and Gradient Boosting models and therefore improving the models themselves. 
    - Then, I used the save_model function to save each model in their own folder within the regression folder.
    - Finally I called the function in my if __name__ == "__main__" block.

### Code Used: *modelling.py*
    
```
def evaluate_all_models():
    np.random.seed(2)
    #specify the serach spaces for hyperparameters in each model
    decision_tree_model = tune_regression_model_hyperparameters("DecisionTreeRegressor", X_train, y_train, X_validation, y_validation, X_test, y_test, hyperpara_searchspace = 
    {
    "criterion" : ["squared_error", "absolute_error"],
    "max_depth" : [20, 25, 30, 35, 40], #initially the set was [10, 20, 30, 40] then I got 30 as an answer so I narrowed down the set.
    "min_samples_split" : np.arange(0.2, 1, 0.2), #initial set = [0.2, 0.4, 2, 4] and got 0.4 
    "max_features" : [6, 7, 8, 9] #initial set = [2, 4, 6, 8] and got 8
    })
    
    print("Evaluation of Decision Tree Model Complete!")
    #save the decision model in a folder called decision_tree
    save_model(decision_tree_model, folder="models/regression/decision_tree")
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
    save_model(random_forest_model, folder="models/regression/random_forest")
    print("Random Forest Model Saved!")
    
    gradient_boosting_model = tune_regression_model_hyperparameters("GradientBoostingRegressor", X_train, y_train, X_validation, y_validation, X_test, y_test, hyperpara_searchspace =
    {
    "learning_rate" : [0.05, 0.1, 0.15], #[0.1, 0.3, 0.5] and got 0.1
    "loss" : ["squared_error", "absolute_error"],
    "n_estimators" : [50, 80, 100], #[20, 50, 100] and got 100
    "max_depth" : [3, 5, 7], 
    "max_features" : [2, 3, 4] #[2, 4, 6] and got 2
    })

    print("Evaluation of Gradient Boosting Model Complete!")
    #save the Gradient Boosting Model in a folder called gradient_boosting
    save_model(gradient_boosting_model, folder="models/regression/gradient_boosting")
    print("Gradient Boosting Model Saved!")

    return decision_tree_model, random_forest_model, gradient_boosting_model

```
7. Find the best overall regression model
    - I defined a find_best_model function to find the best model based on RMSE and return it along with the model's hyperparameters dictionary and performance metrics dictionary.

### Code Used: *modelling.py*

```
def find_best_model(model_details_list):
    
    validation_score = [x[2]["validation_RMSE"] for x in model_details_list]
    linear_performance = evaluate_performance(X_train, y_train, X_test, y_test)
    linear_validation_score = linear_performance[1]
    validation_score.append(linear_validation_score)
    best_model_index = validation_score.index(np.min(validation_score))
    if best_model_index == 3:
        best_model_details = [SGDRegressor(), None, {"validation_MSE" : linear_performance[0], "validation_RMSE": linear_performance[1], "validation_R2": linear_performance[2]}]
    else:
        best_model_details = model_details_list[best_model_index]
    return best_model_details
    
```

# Milestone 5

### **Create a classification model**
There were six tasks in this milestone

1. Create a simple classification model
    - Train a simple classification model

```
def simple_classification_model(X_train, y_train, X_test):
    #Train a simple classification model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    return y_hat

```

2. Evaluate the classification model performance
    - Evaluate the performance of this simple regression model
    - Initially used micro averaging but I got the same value for each

```
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
```

3. Tune the hyperparameters of the models
    - I created a function called tune_classification_model_hyperparameters, which took the agruments: the model class, training, validation, and test sets and a dictionary of hyperparameters.

```
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
```

4. Save the classification model
    - I saved the classification model in a classification folder within the model folder.

```
def save_model(model_details, folder="models/classification/logistic_regression"):
    regression_modelling.save_model(model_details, folder)

```

5. Beat the baseline classification model
    - I improved the performance of the model.

```
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
```
6. Find the best overall classification model
    - 

```
def find_best_model(model_details_list, best_model_indicator, task_folder):
    #adapted the find_best_model in regression_modelling.py to work for a specified best_model_indicator and task_folder.
    best_model_details = regression_modelling.find_best_model(model_details_list, best_model_indicator, task_folder)
    return best_model_details
```

# Milestone 6

### **Create a configurable neural network model**

1. Create the Dataset and Dataloader
    - Created a PyTorch Dataset called AirbnbNightlyPriceImageDataset which returns a tuple (features, label).
    - Then, I created a dataloader for the train set, test set and validation set.
2. Define the first neural network model
    - I defined a PyTorch model class for a fully connected neural network.

3. Create the training loop and train the model

4. Visualise the loss and accuracy of the model

5. Create a configuration file to change the characteristics of the model

6. Save the model

7. Tune the model

I experienced a multiple problems such as a RunTimeError caused the training tensorsbeing different sizes. Another problem I enccountered was exploding gradients. This means that when I trained my model sometimes I would get a Value Error as the y_hat_value would be full of NaN values. Firstly, I increased the batchsizes from 8 to 64 and then I changed the datetype from float32 to float64. 

### Code Used: *neural_network_pricing_model.py*

```
#Create a PyTorch Dataset that returns a tuple (features, label)
class  AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        data = pd.read_csv("/Users/sarahaisagbon/Downloads/clean_tabular_data.csv")
        self.X, self.y = tabular_data.load_airbnb(data, "Price_Night")
        
    def __getitem__(self, idx):
        features = self.X.iloc[idx]
        label = self.y.iloc[idx]
        return (torch.tensor(features), label)
    
    def __len__(self):
        return (len(self.X))

dataset =  AirbnbNightlyPriceImageDataset()

#find the sizes of the training, testing and validation sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
new_train_size = int(0.8 * train_size)
validation_size = train_size - new_train_size

#randomly split the dataset according to the sizes we want
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataset, validation_dataset = random_split(train_dataset, [new_train_size, validation_size])

#Dataloader resulted in [64, 10] at entry 0 and [57, 10] at entry 8
def custom_collate(data): 
    for batch in data:
        inputs, labels = batch
        inputs = [inputs]
        labels = [labels]
        inputs = pad_sequence(inputs, batch_first=True) 
        labels = torch.tensor(labels)
    return inputs, labels
    
#create a Dataloader for the train, test and validation data
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate)

class NNModel(Module):
    def __init__(self, config) -> None:
        super().__init__()
        #Define the layers
        width = config["hidden_layer_width"]
        depth = config["depth"]
        layers = []
        layers.append(torch.nn.Linear(10, width))
        
        for hidden_layer in range(depth):
            if hidden_layer < depth-1:
                #activation functions follwed by 
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Linear(width, width))
            else:
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Linear(width, 1))
                break
            
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, X):
        processed_features = self.layers(X)
        return processed_features
 
def train(model, data_loader, hyper_dict, epochs):
    optimiser_class = hyper_dict["optimiser"]
    optimiser_instance = getattr(torch.optim, optimiser_class)
    optimiser = optimiser_instance(model.parameters(), lr=hyper_dict["learning_rate"])
    
    writer = SummaryWriter()
    
    batch_idx = 0
    
    for epoch in range(epochs):
        for batch in data_loader:
            features, labels = batch
            optimiser.zero_grad()
            #features = torch.unsqueeze(features, 1)
            features = features.to(torch.float32)
            prediction = model(features)
            
            #labels datatype is int64 and everything else has datatype of float64 so I changed the datatype of labels
            labels = labels.to(torch.float32)
            labels = torch.unsqueeze(labels, 1)
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            
            #Optimisation step
            optimiser.step()
            
            correct = (prediction == labels).float().sum()
            total = prediction.size(0)
            accuracy = correct/total
            writer.add_scalar('loss', loss.item(), batch_idx)
            writer.add_scalar('accuracy', accuracy.item(), batch_idx)
            batch_idx += 1

def get_nn_config(yamlfile):
    config = yaml.safe_load(Path(yamlfile).read_text())
    return config
    
def evaluate_model(model, epochs, training_duration):
    #The RMSE loss for training, validation, and test sets
    
    #Get RMSE and R2 for training set
    X_train = torch.stack(tuple(tuple[0] for tuple in train_loader)).type(torch.float32)
    y_train = torch.stack(tuple(tuple[1] for tuple in train_loader)).type(torch.float32)
    y_train = torch.unsqueeze(y_train, 1)
    n = len(y_train)
    
    y_hat_train = model(X_train)
    
    MSE_train = F.mse_loss(y_hat_train, y_train)
    RMSE_train = torch.sqrt(MSE_train)
    #r2_loss didn't work so I calculated R2 from scratch
    #variance = TSS/ n
    TSS_train = torch.var(y_train)
    R2_train = 1 - (MSE_train/TSS_train)
    
    #Get RMSE and R2 for test set
    X_test = torch.stack(tuple(tuple[0] for tuple in test_loader)).type(torch.float32)
    y_test = torch.stack(tuple(tuple[1] for tuple in test_loader)).type(torch.float32)
    y_test = torch.unsqueeze(y_test, 1)
    n = len(y_test)

    y_hat_test = model(X_test)
    
    MSE_test = F.mse_loss(y_hat_test, y_test)
    RMSE_test = torch.sqrt(MSE_test)
    
    TSS_test = torch.var(y_test)
    R2_test = 1 - (MSE_test/TSS_test)

    #Get RMSE and R2 for validation set
    X_validation = torch.stack(tuple(tuple[0] for tuple in validation_loader)).type(torch.float32)
    y_validation = torch.stack(tuple(tuple[1] for tuple in validation_loader)).type(torch.float32)
    y_validation = torch.unsqueeze(y_validation, 1)
    n = len(y_validation)
    
    y_hat_validation = model(X_validation)
    
    MSE_validation = F.mse_loss(y_hat_validation, y_validation)
    RMSE_validation = torch.sqrt(MSE_validation)
    
    TSS_validation = torch.var(y_validation)
    R2_validation = 1 - (MSE_validation/TSS_validation)
    
    RMSE = [RMSE_train.item(), RMSE_test.item(), RMSE_validation.item()]
    R2 = [R2_train.item(), R2_test.item(), R2_validation.item()]
    
    number_of_predictions = epochs * len(train_loader)
    inference_latency = training_duration / number_of_predictions
    
    nn_metrics = {"RMSE_loss": RMSE, "R_squared": R2, "training_duration": training_duration, "inference_latency": inference_latency}
    return nn_metrics

def save_model(model, hyper_dict, nn_metrics, folder="models/regression/neural_networks/"):
    #check that the model is  a PyTorch module.
    if not isinstance(model, torch.nn.Module):
        print("Error: Model is not a PyTorch Module")
    else:
        # Make model folder
        now = datetime.now()
        dt = now.strftime("%d-%m-%y_%H:%M:%S")
        #get the model from the list and save it as joblib file
        model_folder = folder + dt
        
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        torch.save(model.state_dict(), f"{model_folder}/model.pt")
        
        #get the hyperparameters from the list and save it as json file
        hyperparameters = hyper_dict
        with open(f"{model_folder}/hyperparameters.json", "w") as fp:
            json.dump(hyperparameters, fp)
            
        #get the performance from the list and save it as json file
        performance_metrics = nn_metrics
        with open(f"{model_folder}/metrics.json", "w") as fp:
            json.dump(performance_metrics, fp)

def do_full_model_train(hyper_dict, epochs=5):
    model = NNModel(hyper_dict)
    start_time = time.time()
    train(model, train_loader, hyper_dict, epochs)
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"It took {training_duration} seconds to train the model")
    metrics_dict = evaluate_model(model, training_duration, epochs)
    save_model(model, hyper_dict, metrics_dict)
    print(hyper_dict)
    model_info = [model, hyper_dict, metrics_dict]
    return model_info

def generate_nn_configs():
    search_space = {
    #Adam is generally better as faster computation time, and require fewer parameters for tuning
    #But it performs worse than SGD for image classification
    "optimiser": ["SGD", "Adam"],
    "learning_rate": [0.001, 0.002],
    "hidden_layer_width": [5, 10],
    "depth": [2, 3]
    }
    
    keys, values = zip(*search_space.items())
    #get all the combinations of the keya
    hyper_value_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    return hyper_value_dict_list

def find_best_nn(epochs):
    lowest_RMSE_loss_validation = np.inf
    hyper_value_dict_list = generate_nn_configs()
    for hyper_value_dict in hyper_value_dict_list:
        model_info = do_full_model_train(hyper_value_dict, epochs)
        nn_metrics = model_info[2]
        #I used RMSE loss to find the best model
        RMSE_loss = nn_metrics["RMSE_loss"]
        RMSE_loss_validation = RMSE_loss[1]
        
        if RMSE_loss_validation < lowest_RMSE_loss_validation:
            lowest_RMSE_loss_validation = RMSE_loss_validation
            best_model_info = model_info

        time.sleep(1)
    best_model, best_hyper_dict, best_nn_metrics = best_model_info
    print(best_model, best_hyper_dict, best_nn_metrics)
    
    save_model(best_model, best_hyper_dict, best_nn_metrics, "models/regression/neural_networks/best_neural_networks")

    
if __name__ == "__main__":
    find_best_nn(5)

```

