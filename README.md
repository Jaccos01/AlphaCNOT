
# AlphaCNOT repository
A very basic implementation of AlphaZero style learning with the MCTX framework for Monte Carlo Tree Search in JAX. The included basic_tree_search.py script assumes the agent have access to the exact environment dynamics and does not include any model learning, but rather learns a policy and value function from tree search results as in AlphaZero.
## ---->CHANGE

## Dependencies
Using this repo requires JAX, Haiku, NumPy and wandb...



## Training a model
You can start a training of a new AlphaCNOT model with 
```bash
# python train_model --n_qubits "your_card" --topology "your_topology" --reward "reward_style" --steps "your_training_steps"
python train_model --n_qubits "your_card" --topology "your_topology" --steps "your_training_steps"
```

## Validating a model
You can validate a model of AlphaCNOT with 
```bash
python test_model --n_qubits "your_card" --topology "your_topology" --name "name_of_the_model" --s "number_of_matrices" --r "number_of_repetitions_for_each_matrix"
```
In case you select r>1, the model will find multiple syntheses for each matrix and eventually will choose the shortest one for the final statistics.

## Synthesize your custom circuit
Use "synthesize_circuit.ipynb"
    1) Load a model
    2) Create a CNOT circuit
    3) Automatic matrix translation
    4) AlphaCNOT synthesis 

