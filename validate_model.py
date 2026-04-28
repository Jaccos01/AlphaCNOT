import argparse
import os
import json

from scripts import run_validation 
from scripts.utils import parse_test_args 


def main():
    # 1. PARSE ARGUMENTS
    args = parse_test_args()

    # 2. MODEL PATHS
    model_name = f"model_AlphaCNOT_{args.n_qubits}Q_{args.topology}_top"
    model_folder = f"models/Card_{args.n_qubits}/{model_name}"
    
    params_path = os.path.join(model_folder, f"{model_name}.params")
    config_file = os.path.join(model_folder, f"{model_name}.json")

    # 3. LOAD CONFIGURATION
    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        return

    with open(config_file, 'r') as f:
        config = json.load(f)

    print(f"--- Launching Validation ---")
    print(f"Target Folder: {model_folder}")
    print(f"Testing on {args.test} matrices with {args.rep} repetitions each.")

    # 4. RUN VALIDATION
    run_validation(
        config=config,
        params_path=params_path,
        model_folder=model_folder,
        n_test=args.test,
        n_rep=args.rep,
        top=args.topology
    )

    print(f"--- Validation Process Finished ---")

if __name__ == "__main__":
    main()