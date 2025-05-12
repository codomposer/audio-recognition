import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import time

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(.2),
            nn.Flatten(),
            nn.Linear(in_features=12288, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def eval_model(model, data_loader):
    """
    Evaluate the model's performance on a given data loader
    
    Args:
        model: The trained model to evaluate
        data_loader: DataLoader containing the evaluation data
        
    Returns:
        accuracy: The model's accuracy on the data
        test_loss: The total loss on the data
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    test_loss = 0
    loss_fn = nn.BCELoss()
    
    with torch.no_grad():
        for x, y in data_loader:
            # Forward pass
            outputs = model(x)
            
            # Get predictions
            y_pred = torch.round(outputs)
            y_pred_list.extend(y_pred.clone().detach().tolist())
            y_true_list.extend(y.clone().detach().tolist())
            
            # Get the loss
            loss = loss_fn(outputs, y.reshape(-1, 1).to(torch.float32))
            
            # Keep a running total
            test_loss += loss.item()
    
    acc = classification_report(y_true_list, y_pred_list, output_dict=True, zero_division=0)['accuracy']
    return acc, test_loss

def train_model(model, train_loader, test_loader, epochs=50, lr=0.0001):
    """
    Train the model
    
    Args:
        model: The model to train
        train_loader: DataLoader containing the training data
        test_loader: DataLoader containing the testing/validation data
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        model: The trained model
        training_stats: Dictionary containing training statistics
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        model = model.to(device)
    
    # Loss function and optimizer
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Track metrics
    epoch_nums = []
    training_loss = []
    validation_loss = []
    validation_acc = []
    
    # Track best model
    current_best = 0
    best_model = None
    
    start = time.time()
    
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        
        for x, y in train_loader:
            # Move data to device if available
            x = x.to(device)
            y = y.to(device)
            
            # Reset the optimizer
            optimizer.zero_grad()
            
            # Forward pass
            output = model(x)
            
            # Get the loss
            loss = loss_fn(output, y.reshape(-1, 1).to(torch.float32))
            
            # Keep a running total
            train_loss += loss.item()
            
            # Backpropagate
            loss.backward()
            optimizer.step()
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            metric, test_loss = eval_model(model, test_loader)
            if metric > current_best:
                best_model = model
                current_best = metric
                print(f'best accuracy so far is {current_best}')
            
            epoch_nums.append(epoch)
            training_loss.append(train_loss)
            validation_loss.append(test_loss)
            validation_acc.append(metric)
    
    end = time.time()
    training_time = (end - start) / 60
    
    print(f"{training_time} minutes to train")
    
    # Return the best model and training statistics
    training_stats = {
        'epoch_nums': epoch_nums,
        'training_loss': training_loss,
        'validation_loss': validation_loss,
        'validation_acc': validation_acc,
        'best_accuracy': current_best,
        'training_time': training_time
    }
    
    return best_model, training_stats
