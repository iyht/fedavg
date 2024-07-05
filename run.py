import json
import subprocess
import signal
import sys
import os
import datetime
import time
from time import sleep
import matplotlib.pyplot as plt

def plot_accuracy(data, baseline, output_file):
    plt.figure(figsize=(10, 6))

    for file_name, accuracy in data:
        rounds, accuracies = zip(*accuracy)
        plt.plot(rounds, accuracies, label=file_name)

    plt.axhline(y=baseline, color='r', linestyle='--', label='Centralized Training')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title('Communication Rounds vs. Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def main():
    processes = []
    all_data = []
    all_configs = ["config_B10_E1_non_iid", "config_B10_E1_iid", "config_Bfull_E1_iid", "config_Bfull_E1_non_iid"]

    try:
        prefix = time.strftime("eval%s")
        os.makedirs(prefix)
        os.environ["PREFIX_PATH"] = prefix

        centralized_config_path = "centralized.json"
        centralized_proc = subprocess.Popen(["python mnist.py --train-conf {}".format(centralized_config_path)], shell=True, stdout=subprocess.PIPE, env=os.environ)
        # processes.append(centralized_proc)

        for fed_config_name in all_configs:
            processes = []
            fed_config_path = fed_config_name + ".json"
            with open(fed_config_path, 'r') as file:
                fed_config = json.load(file)

            with open(os.path.join(prefix, fed_config_name + "_server.log"), "w") as f:
                server_proc = subprocess.Popen(["python server.py --train-conf {}".format(fed_config_path)], shell=True, stdout=f, stderr=f, env=os.environ)
                processes.append(server_proc)
                sleep(3)

            for i in range(0, fed_config["num_clients"]):
                with open(os.path.join(prefix, fed_config_name + "_client_{}.log".format(i)), "w") as f:
                    p = subprocess.Popen(["python client.py --train-conf {} --partition-id {}".format(fed_config_path, i)], shell=True, env=os.environ, stdout=f, stderr=f)
                    processes.append(p)

            for p in processes:
                p.wait()
        
            server_result_path = os.path.join(prefix, fed_config_name+"_server_result.json")
            with open(server_result_path, 'r') as f:
                data = json.load(f)
                all_data.append((fed_config["description"], data['accuracy']))
        
        centralized_proc.wait()
        with open(os.path.join(prefix, "centralized_result.json"), "r") as f:
            data = json.load(f)
            baseline = data['accuracy'][-1][1]
        plot_accuracy(all_data, baseline, os.path.join(prefix, "centralized_fedavg_accuracy_comparison.png"))
        
        
        
    except KeyboardInterrupt:
        print("Terminating all subprocesses...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()  # Ensure all processes have terminated
        sys.exit(1)

if __name__ == "__main__":
    main()