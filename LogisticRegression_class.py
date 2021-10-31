
import numpy as np

class LogisticRegression(object):
    
    def __init__(self,input_size):
        self.W = np.random.randn(input_size,1)
    
    def sigmoid(self,z):
        h = 1 / (1 + np.exp(-z))
        return h
    
    def train(self,X,Y,lr,epochs=100):
        loss_history = []
        
        for i in range(epochs):
            loss = self.loss(X,Y)
            loss_history.append(loss)
            
            dW = self.back_propagation(X,Y)
            self.W = self.W - lr * dW
            
        return loss_history
        
    def loss(self,X,Y):
        z = np.dot(X,self.W)
        h = self.sigmoid(z)
        
        n = X.shape[0]
        loss = np.abs(1/n * np.sum(-Y * np.log(h) - (1-Y) * np.log(1-h)))
        
        return loss
    
    def back_propagation(self,X,Y):
        z = -Y * np.dot(X,self.W)
        n = X.shape[0]
        dW = -1/n * np.sum((Y * X * self.sigmoid(z)) , axis=0)
        dW = dW.reshape(-1,1)
        
        return dW
    
    def predict(self,X):
        z = np.dot(X,self.W)
        y_hat = self.sigmoid(z)
        Y_pred = np.where((y_hat > 0.5),1,-1)  # threshold
        
        return Y_pred
    