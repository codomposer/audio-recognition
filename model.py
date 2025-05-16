import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import time
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,      # e.g., number of mel-frequency bins (like 40)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0  # Apply dropout between LSTM layers
        )
        
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * direction_factor, 128),
            nn.BatchNorm1d(128),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),   # Add batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, time_steps, input_size)
        _, (h_n, _) = self.lstm(x)
        # If bidirectional, concatenate last forward and backward hidden states
        if self.lstm.bidirectional:
            h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            h_n = h_n[-1,:,:]
        return self.classifier(h_n)

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

def train_model(model, train_loader, test_loader, epochs=60, lr=0.001, weight_decay=1e-5):
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
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Add L2 regularization
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',        # Monitor validation accuracy
        factor=0.5,       # Reduce LR by half when plateau
        patience=5,       # Wait 5 epochs before reducing LR
        min_lr=1e-6       # Minimum learning rate
    )
    print(f"Learning rate scheduler initialized with patience={5}")
    
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
        
        # Evaluate after each epoch
        metric, test_loss = eval_model(model, test_loader)
        
        # Update learning rate scheduler
        scheduler.step(metric)
        
        # Save best model
        if metric > current_best:
            # Create a deep copy of the model
            best_model = type(model)(
                input_size=model.lstm.input_size,
                hidden_size=model.lstm.hidden_size,
                num_layers=model.lstm.num_layers,
                bidirectional=model.lstm.bidirectional
            )
            best_model.load_state_dict(model.state_dict())
            best_model = best_model.to(device)
            current_best = metric
            print(f'Epoch {epoch}: New best accuracy: {current_best:.4f}')
        
        # Record training statistics
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
        validation_acc.append(metric)
        
        # Print epoch statistics
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {test_loss:.4f}, Val Acc: {metric:.4f}')
    
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
