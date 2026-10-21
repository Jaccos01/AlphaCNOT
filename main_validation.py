import os
import json
from scripts.run_validation import run_validation
from scripts.model_loading import resolve_trained_model_path
from scripts.utils import load_alphacnot_model, parse_test_args

def main():
    args = parse_test_args()

    if args.source == "trained_models":
        model_path = resolve_trained_model_path(
            args.n_qubits,
            args.topology,
            args.reward,
            args.terminal_mode,
        )
        model_bundle = load_alphacnot_model(
            args.n_qubits,
            args.topology,
            source="trained_models",
            reward_type=args.reward,
            terminal_mode=args.terminal_mode,
        )
        run_validation(
            config=None,
            params_path=None,
            model_folder=str(model_path.with_suffix("")),
            n_test=args.test,
            n_rep=args.rep,
            top=args.topology,
            model_bundle=model_bundle,
        )
        return

    reward_label = args.reward.upper()
    model_base_name = f"model_AlphaCNOT_{args.n_qubits}Q_{args.topology}_top"
    model_folder = f"models/Card_{args.n_qubits}/{model_base_name}_{reward_label}"
    params_path = os.path.join(model_folder, "params", f"{model_base_name}.params")
    config_file = os.path.join(model_folder, f"{model_base_name}.json")

    if not os.path.exists(config_file):
        print(f"Error: Config file not found in {config_file}")
        return
    if not os.path.exists(params_path):
        print(f"Error: Params file not found in {params_path}")
        return

    with open(config_file, 'r') as f:
        config = json.load(f)

    run_validation(
        config=config,
        params_path=params_path,
        model_folder=model_folder,
        n_test=args.test,
        n_rep=args.rep,
        top=args.topology,
    )

if __name__ == "__main__":
    main()
