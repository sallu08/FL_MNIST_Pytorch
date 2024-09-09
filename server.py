import random
from typing import Dict, Optional, Tuple
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm  # Progress bar

# Set random seed for reproducing exact result each time
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# GLobal model 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

def main() -> None:
    # Set random seed
    set_random_seed(42)
    
    # Check for CUDA availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and compile model
    model = CNN().to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,              # Proportion of clients to be used for training
        fraction_evaluate=0.3,         # Proportion of clients sampled for evaluation randomly
        min_fit_clients=3,             # Number of active clients required for training
        min_evaluate_clients=3,        # Number of active clients required for evaluation
        min_available_clients=3,       # Minimum number of clients needed 
        evaluate_fn=get_evaluate_fn(model, device),  
        on_fit_config_fn=fit_config,         
        on_evaluate_config_fn=evaluate_config,  
        initial_parameters=fl.common.ndarrays_to_parameters([p.data.cpu().numpy() for p in model.parameters()]),
    )

    # Start Flower server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3), # One round is completed when all clients have run local model 3 times

        strategy=strategy,
    )

def get_evaluate_fn(model, device):
    """Return an evaluation function for server-side evaluation."""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    #Evaluation is performed on all 60,000 images of MNIST
    val_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform) 
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.load_state_dict({name: torch.tensor(param) for name, param in zip(model.state_dict().keys(), parameters)})
        model.to(device)
        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        # Progress bar for evaluation
        with tqdm(total=len(val_loader), desc="Evaluating", unit="batch") as pbar: #show all code below in form of progress-bar in cmd
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    pbar.update(1)

        accuracy = correct / total
        return val_loss / len(val_loader), {"classification accuracy on server": accuracy}

    return evaluate

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    return {
        "batch_size": 32,        # 32 images at a time are taken as input
        "local_epochs": 3        # Each client runs model (3) times then sends result to server
    }

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    return {"val_steps": 5}      
if __name__ == "__main__":
    main()

