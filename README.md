
# AlphaCNOT: Learning CNOT Minimization with Model-Based Planning
A very basic implementation of AlphaZero style learning with the MCTX framework for Monte Carlo Tree Search in JAX. The training script model_training.py (based on "basic_tree_search.py script from https://github.com/kenjyoung/mctx_learning_demo) assumes the agent have access to the exact environment dynamics and does not include any model learning, but rather learns a policy and value function from tree search results as in AlphaZero.

## Dependencies
Using this repo requires JAX, Haiku, NumPy


## Training a model
You can start a training of a new AlphaCNOT model with 
```bash
python train_model --n_qubits "your_card" --topology "your_topology" --reward "reward_type" --epochs "your_training_epochs"
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

# Citation
@article{cossio2026alphacnot,
  title={AlphaCNOT: Learning CNOT Minimization with Model-Based Planning},
  author={Cossio, Jacopo and Bosco, Daniele Lizzio and Romanello, Riccardo and Serra, Giuseppe and Piazza, Carla},
  journal={arXiv preprint arXiv:2604.13812},
  year={2026}
}