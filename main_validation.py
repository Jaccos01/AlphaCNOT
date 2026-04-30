import argparse
import os
import json
from scripts.run_validation import run_validation
from scripts.utils import parse_test_args

def main():
    args = parse_test_args()

    reward_label = args.name.upper() if args.name in ["mix", "inf"] else "MIX"
    
    model_base_name = f"model_AlphaCNOT_{args.n_qubits}Q_{args.topology}_top"
    model_folder = f"models/Card_{args.n_qubits}/{model_base_name}_{reward_label}"

    params_path = os.path.join(model_folder, "params", f"{model_base_name}.params")
    config_file = os.path.join(model_folder, f"{model_base_name}.json")

    if not os.path.exists(config_file):
        print(f"❌ Errore: Configurazione non trovata in {config_file}")
        return
    if not os.path.exists(params_path):
        print(f"❌ Errore: Pesi non trovati in {params_path}")
        return

    with open(config_file, 'r') as f:
        config = json.load(f)

    run_validation(
        config=config,
        params_path=params_path,
        model_folder=model_folder,
        n_test=args.test,
        n_rep=args.rep,
        top=args.topology
    )

if __name__ == "__main__":
    main()