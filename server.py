"""Flower server example."""

from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import os
import json


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    res = {"accuracy": sum(accuracies) / sum(examples)}
    return res


# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Start Flower server
history = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=strategy,
)

prefix = os.getenv("PREFIX_PATH")
if prefix == None: prefix = "./"
result_path = os.path.join(prefix, "server_result.json")
json.dump(history.metrics_distributed, open(result_path, 'w' ) )