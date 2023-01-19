import numpy as np

class LinearRegression:
    def __init__(self, optimiser, n_features, num_jobs, learning_rate):
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()
        self.num_jobs = num_jobs
        self.lr = learning_rate
        
    #get predicted values
    def predict(self, X):
        self.yhat = X.dot(self.w) + self.b
        return self.yhat
    
    #evaluate the loss
    def MSE(self, yhat, y):
        self.errors = yhat - y
        squared_errors = self.errors ^ 2
        loss = sum(squared_errors)/len(squared_errors)
        return loss
    
    def _calc_deriv(self, X, yhat, y):
        m = len(y) ## m = number of examples
        dLdw = 2 * np.sum(X.T * self.errors).T / m ## calculate the loss derivative with respect to the weights
        dLdb = 2 * np.sum(self.errors) / m ## calculate the loss derivative with respect to the bias
        return dLdw, dLdb ## return the rate of change in the loss with respect to w and b, separately.
    
    def step(self, w, b, X, yhat, y):
        dLdw, dLdb = self._calc_deriv(X, yhat, y)
        new_w = w - self.lr * dLdw
        new_b = b - self.lr * dLdb
        return new_w, new_b
    
    def _update_params(self, new_w, new_b):
        self.w = new_w ## set this instance's weights to the new weight value passed to the function
        self.b = new_b ## do the same for the bias
        
    #optimise the parameters
    def fit(self, X,y):
        all_costs = []
        for job in self.num_jobs:
            # MAKE PREDICTIONS AND UPDATE MODEL
            predictions = self.predict(X) ## make predictions
            new_w, new_b = self.step(self.w, self.b, X, predictions, y) ## calculate updated params
            self._update_params(new_w, new_b) ## update the model weight and bias
            
            # CALCULATE THE LOSS FOR VISUALISATION
            cost = self.MSE(predictions, y) ## compute the loss 
            all_costs.append(cost) ## add the cost for this batch of examples to the list of costs (for plotting)
            
if __name__ == "__main__":
    model = LinearRegression()
    model.fit