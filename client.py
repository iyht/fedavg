import argparse
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import mnist
import flwr as fl


USE_FEDBN: bool = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Flower Client
class MNISTClient(fl.client.NumPyClient):

    def __init__(
        self,
        model: mnist.Net,
        train_loader: DataLoader,
        test_loader: DataLoader,
        conf: Dict[str, Any]
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_conf = conf

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding
            # parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        mnist.train(self.model, self.train_loader, self.test_loader, self.train_conf)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        print("Client Evaluation: ")
        self.set_parameters(parameters)
        loss, accuracy = mnist.test(self.model, self.test_loader, self.train_conf)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}


def main() -> None:
    """Load data, start MNISTClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition-id", type=int, required=True)
    parser.add_argument('--train-conf', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    train_conf = mnist.load_train_conf(args.train_conf)

    # Load data
    train_loader, test_loader = mnist.load_data(args.partition_id, train_conf)

    # Load model
    model = mnist.Net().to(train_conf["device"]).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(train_loader))[0].to(train_conf["device"]))

    # Start client
    client = MNISTClient(model, train_loader, test_loader, train_conf).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
