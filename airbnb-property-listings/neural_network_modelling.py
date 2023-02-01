import numpy as np
import pandas as pd
import tabular_data
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module
import yaml
from pathlib import Path
import os
import json
from datetime import datetime
import time
import itertools
from sklearn.metrics import r2_score


#Create a PyTorch Dataset that returns a tuple (features, label)
class  AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("/Users/sarahaisagbon/Downloads/clean_tabular_data.csv")
        self.X, self.y = tabular_data.load_airbnb(self.data, "Price_Night")
        
    def __getitem__(self, idx):
        features = self.X.iloc[idx]
        label = self.y.iloc[idx]
        return (torch.tensor(features), label)
    
    def __len__(self):
        return (len(self.data))

dataset =  AirbnbNightlyPriceImageDataset()

#find the sizes of the training, testing and validation sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
new_train_size = int(0.8 * train_size)
validation_size = train_size - new_train_size

#randomly split the dataset according to the sizes we want
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataset, validation_dataset = random_split(train_dataset, [new_train_size, validation_size])

#create a Dataloader for the train, test and validation data
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True)


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
            features = features.to(torch.float64)
            prediction = model(features)
            
            #loss = F.mse_loss(prediction, labels)
            #labels size is [16] but we have inputs with size [16,1] so we change the size of labels to [16, 1]
            labels = torch.unsqueeze(labels, 1)
            
            #labels datatype is int64 and everything else has datatype of float64 so I changed the datatype of labels
            labels = labels.to(torch.float64)
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
    X_train = torch.stack([tuple[0] for tuple in train_loader]).type(torch.float64)
    y_train = torch.stack([torch.tensor(tuple[1]) for tuple in train_loader])
    y_train = torch.unsqueeze(y_train, 1)
    y_train = y_train.float()
    y_hat_train = model(X_train)
    RMSE_train = torch.sqrt(F.mse_loss(y_hat_train, y_train))
    y_hat_train = y_hat_train.detach().numpy()
    #r2_loss didn't work so I calculated R2 from scratch
    R2_train = r2_score(y_train, y_hat_train)
    
    #Get RMSE and R2 for test set
    X_test = torch.stack([tuple[0] for tuple in test_loader]).type(torch.float64)
    y_test = torch.stack([torch.tensor(tuple[1]) for tuple in test_loader])
    y_test = torch.unsqueeze(y_test, 1)
    y_test = y_test.float()
    y_hat_test = model(X_test)
    RMSE_test = torch.sqrt(F.mse_loss(y_hat_test, y_test))
    R2_test = r2_score(y_test, y_hat_test)

    
    #Get RMSE and R2 for validation set
    X_validation = torch.stack([tuple[0] for tuple in validation_loader]).type(torch.float64)
    y_validation = torch.stack([torch.tensor(tuple[1]) for tuple in validation_loader])
    y_validation = torch.unsqueeze(y_validation, 1)
    y_validation = y_validation.float()
    y_hat_validation = model(X_validation)
    RMSE_validation = torch.sqrt(F.mse_loss(y_hat_validation, y_validation))
    R2_validation = r2_score(y_validation, y_hat_validation)
    
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
    "learning_rate": [0.001, 0.005],
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

    