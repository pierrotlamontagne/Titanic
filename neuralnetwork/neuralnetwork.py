import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class NeuralNetwork(nn.Module): 
    def __init__(self): 
        super(NeuralNetwork, self).__init__() #Inherit from nn.Module
        #Activation function
        self.activation = nn.ReLU() #Rectified Linear Unit
        
        #Linear process
        self.linear_stack = nn.Sequential(
            nn.Linear(11, 120),
            self.activation,
            nn.Linear(120, 84),
            self.activation,
            nn.Linear(84, 37),
            self.activation,
            nn.Linear(37,1)
        )
        self.sigmoid = nn.Sigmoid() #Sigmoid function
        
    #Forward propagation
    def forward(self, x):
        x = self.linear_stack(x)
        output = self.sigmoid(x)
        
        return output


def train_loop(dataloader, model, loss_fn, optimizer, lr_sch=False):
    
    size = len(dataloader.dataset) #Number of images   

    train_loss = 0.0 #Loss
    
    for batch, (X,y,Id) in enumerate(dataloader):
        #X = xy[0]
        #y = xy[1]
        pred = model(X) # Compute prediction error
        loss = loss_fn(pred, y) # Backpropagation

        optimizer.zero_grad() #Reset the gradients
        loss.backward() #Compute the gradients  
        optimizer.step() #Update the weights  
        
        # if lr_sch==True: #If a learning rate scheduler is used
        #     lr_scheduler.step() #Update the learning rate
        
        loss, current = loss.item(), batch * len(X) #Loss and current number of images
        train_loss += loss * X.size(0) #Add the loss to the total loss
        
        if batch % 100 == 0: #Print the loss every 100 batches
            print(f"Loss: {loss}, [{current}/{size}]") 

    return train_loss / size

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    # On se sauve les gradients comme ils ne sont pas utilisés
    with torch.no_grad():
        for xy in dataloader:
            X = xy[0]
            y = xy[1]
            pred = model(X)
            test_loss += loss_fn(pred, y).item()  # Compute loss

    test_loss /= num_batches
    RMSE = np.sqrt(test_loss)
    print(f"Avg loss: {test_loss} \n")
    print("RMSE = {}".format(RMSE))

    return test_loss