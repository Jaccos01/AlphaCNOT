import argparse
import os
import json

from scripts import model_training 
from scripts.utils import parse_train_args


def main():
    args = parse_train_args()

    # 1. LOAD BASE CONFIGURATION
    config_path = "scripts/config_train.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 2. UPDATE CONFIG WITH CLI INPUTS
    config["env_config"]["cardinality"] = args.n_qubits
    config["env_config"]["topology"] = args.topology
    config["num_steps"] = args.epochs

    model_name = f"model_AlphaCNOT_{args.n_qubits}Q_{args.topology}_top"
    # 3. DEFINE OUTPUT PATHS
    output_dir = f"models/Card_{args.n_qubits}/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Launching Training ---")
    print(f"Qubits: {args.n_qubits} | Topology: {args.topology} | Reward: {args.r} | Steps: {args.epochs}")

    model_training(
        model_name=model_name,
        n_qubits=args.n_qubits,
        topology=args.topology,
        reward = args.r,
        config=config,
        epochs=args.epochs,
        output_dir=output_dir
    )

    print(f"--- Training Process Finished ---")

if __name__ == "__main__":
    main()
