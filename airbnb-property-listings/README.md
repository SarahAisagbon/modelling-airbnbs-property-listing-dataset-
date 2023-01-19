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
3. Tune the hyperparameters of the models

4. Save the classification model

5. Beat the baseline classification model

6. Find the best overall classification model