import os
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import Subset

# Make PyTorch logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Define the Austrailia Campus CNN model
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

# Define Flower client
class My_Client(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()

        # Train the model
        self.model.train()
        for epoch in range(epochs):
            for batch in self.train_loader:
                x_train, y_train = batch
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(x_train)
                loss = criterion(outputs, y_train)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

        # Return updated model parameters and results
        parameters_prime = self.get_parameters()
        num_examples_train = len(self.train_loader.dataset)
        results = {"loss": loss.item()}
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.set_parameters(parameters)

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Evaluate the model
        self.model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for x_test, y_test in self.test_loader:
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                outputs = self.model(x_test)
                loss += criterion(outputs, y_test).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_test).sum().item()

        num_examples_test = len(self.test_loader.dataset)
        accuracy = correct / num_examples_test
        return loss, num_examples_test, {"accuracy": accuracy}

    def set_parameters(self, parameters):
        """Set local model parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

def load_partition():
    """Load a subset of MNIST data to simulate a partition."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the dataset
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)   #total 60000 images
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)   #total 10000 images
    train_data = Subset(train_data, indices=list(range(0, 5000)))  # 5000 samples for training set
    test_data = Subset(test_data, indices=list(range(0, 1000)))      # 1000 samples for test set


    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, test_loader

def main() -> None:

    # Load and compile the PyTorch model
    model = CNN()

    # Load a subset of data to simulate the local data partition
    train_loader, test_loader = load_partition()

    # Start Flower client
    client = My_Client(model, train_loader, test_loader)

    fl.client.start_client(
        server_address="localhost:8080",
        client=client,
        # root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )

if __name__ == "__main__":
    main()
