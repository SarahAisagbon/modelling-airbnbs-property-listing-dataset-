# **Modelling Airbnb's property listing dataset**

Building a framework that systematically trains, tunes, and evaluates models on several tasks that are tackled by the Airbnb team
 - Cleaned and visualised data (Pandas)
 - Performed feature selection to determine relationships between features and target columns.
 - Trained, compared and evaluated machine learning models (Random Forest, Linear/Logistic Regression, XGBoost etc) for classification (determining different Airbnb categories) & regression (predicting tariff) use cases.
 - Performed hyperparameter tuning and cross validation to optimise the results for particular metrics, such as precision in classification.

# Project Documentation

## Milestone 1: Data Preparation
Technologies / Skills:
- Pandas
- OS
- Pillow
- Joblib
- JSON

### **1. Cleaning tabular data**
 - Cleaned the tabular data by removing missing ratings, formatting the descriptions and replacing na values in the feature column.
 - Saved cleaned data in [clean_tabular_data.csv](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/blob/main/airbnb-property-listings/tabular_data/.clean_tabular_data.csv)
 - The data was returned as tuples (features and labels) to be used by a machine learning model
 - Can be found in [tabular_data.py](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/blob/main/airbnb-property-listings/tabular_data.py) file

### **2. Preparing the image data**
- Checked that all images were RGB
- Resized all images to the same height as the smallest image.
- Saved them in [processed_images](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/tree/main/airbnb-property-listings/processed_images) folder
- Can be found in [prepare_image_data.py](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/blob/main/airbnb-property-listings/prepare_image_data.py) file

## Milestone 2: Regression Model
Technologies / Skills:
- sklearn

### **1. Training a simple regression model**
- Imported load_airbnb function from [tabular_data.py](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/blob/main/airbnb-property-listings/tabular_data.py)
- Created a linear regression model using SGDRegressor model class
- Can be found in the [regression_price_model.py](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/blob/main/airbnb-property-listings/regression_price_model.py) file

### **2. Evaluating the performance**
- Compute  and use the performance metrics: MSE, RMSE and R2 to evaluate the performance

### **3. Implementing a custom function to tune the hyperparemeters of the model**
- Created the custom_tune_regression_model_hyperparameters function to implement the Grid Search method on the hyperparameters of a linear regression model, a decision tree model, a random forest model and a gradiant boosting model
- Function returned the best model, a dictionary of the best hyperparameters for the model and a dictionary of performance metrics

### **4. Tuning the hyperparameters using the GridSearchCV function in sklearn**
- This returned the same list as when the custom function was used

### **5. Saving the model**
- Created a save_model function to save the model, hyperparameter dictionary and performance metric dictionary in the [models](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/tree/main/airbnb-property-listings/models) folder

### **6. Beating the baseline regression model**
- Defined the evaluate_all_models function, which uses the tune_regression_model_hyperparameters function to improve the hyperparameters of the Decision Tree, Random Forest and Gradient Boosting models and the models themselves. 
- Used the save_model function to save each model in their own folder within the [regression](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/tree/main/airbnb-property-listings/models/regression) folder.

### **7. Finding the best overall regression model**
- Defined the find_best_model function to find the best model based on RMSE and return it along with the model's hyperparameters dictionary and performance metrics dictionary.

## Milestone 3: Classification Model
Technologies / Skills:
- sklearn

### **1. Training a simple classification model**
- Imported load_airbnb function from [tabular_data.py](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/blob/main/airbnb-property-listings/tabular_data.py)
- Trained a simple classification model using SGDRegressor model class
- Can be found in the [classification_modelling.py](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/blob/main/airbnb-property-listings/classification_modelling.py) file

### **2. Evaluating the performance**
- Computed and used the performance metrics: MSE, RMSE and R2 to evaluate the performance
- Initially, I used micro averaging but I got the same value for each

### **3. Tuning the hyperparameters using the GridSearchCV function in sklearn**
- Created the tune_classification_model_hyperparameters function to implement the Grid Search method on the hyperparameters of a logistic regression model, a decision tree model, a random forest model and a gradiant boosting model

### **4. Saving the model**
- Created a save_model function, which uses the save_model function from the [regression_price_model.py](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/blob/main/airbnb-property-listings/regression_price_model.py) file to save the model, hyperparameter dictionary and performance metric dictionary in the [models](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/tree/main/airbnb-property-listings/models) folder within the [classification](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/tree/main/airbnb-property-listings/models/classification) subfolder

### **5. Beating the baseline regression model**
- I improved the performance of the model.

### **7. Finding the best overall regression model**
- I adapted the find_best_model function from the [regression_price_model.py](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/blob/main/airbnb-property-listings/regression_price_model.py) file

## Milestone 4: Configurable Neural Network Model
Technologies / Skills:
- Torch
- YAML
- OS

### **1. Creating the Dataset and Dataloader**
- Created a PyTorch Dataset called AirbnbNightlyPriceImageDataset which returns a tuple (features, label).
- Created a dataloader for the train set, test set and validation set.
- Can be found in [neural_network_pricing_model.py](https://github.com/SarahAisagbon/modelling-airbnbs-property-listing-dataset-/blob/main/airbnb-property-listings/neural_network_pricing_model.py)

### **2. Defining the first neural network model**
- Defined a PyTorch model class for a fully connected neural network.
### **3. Create the training loop and train the model**
### **4. Visualise the loss and accuracy of the model**
### **5. Create a configuration file to change the characteristics of the model**
### **6. Save the model**
### **7. Tune the model**

### *Issues:*
1. RunTimeError because the training tensors was different sizes.
*Solution: Increased the batchsizes from 8 to 64*
2.Value Error because of exploding gradients. This means that when I trained my model sometimes I would get a Value Error as the y_hat_value would be full of NaN values.
*Solution: Changed the datetype from float32 to float64*

