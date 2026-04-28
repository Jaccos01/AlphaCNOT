import jax
import jax.numpy as jnp
import haiku as hk
import optax    
import chex
import os
import pickle as pkl
import json
import mctx
import functools
from scripts import env
import argparse

def parse_train_args():
    parser = argparse.ArgumentParser(description="Circuit Optimization Training CLI")
    
    # CLI Arguments
    parser.add_argument("--n_qubits", type=int, default=5, help="Number of qubits (cardinality)")
    parser.add_argument("--topology", type=str, default="ALL", help="Network topology (e.g., ALL, L, T, Y)")
    #parser.add_argument("--reward", type=str, default="mix", help="Reward function type")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training steps/epochs")
    
    return parser.parse_args()

def parse_test_args():
    parser = argparse.ArgumentParser(description="Circuit Optimization Testing/Validation CLI")
    
    # CLI Arguments
    parser.add_argument("-n_qubits", type=int, default=5, help="Number of qubits (cardinality)")
    parser.add_argument("-topology", type=str, default="ALL", help="Topology used (e.g., ALL, L, T, Y)")
    parser.add_argument("-name", type=str, default="None", help="The name of the model folder")
    parser.add_argument("-test", type=int, default=100, help="Number of random matrices to validate")
    parser.add_argument("-rep", type=int, default=10, help="Number of repetitions per matrix")
    
    return parser.parse_args()

activation_dict = {"relu": jax.nn.relu, "silu": jax.nn.silu, "elu": jax.nn.elu}

class V_function(hk.Module):
        def __init__(self, config, name=None):
            super().__init__(name=name)
            self.num_hidden_units = config["num_hidden_units"]
            self.num_hidden_layers = config["num_hidden_layers"]
            self.activation = activation_dict[config["activation"]]

        def __call__(self, obs):
            x = jnp.ravel(obs)

            identity1 = hk.Linear(self.num_hidden_units)(x) 
            x = self.activation(identity1)
            x = self.activation(hk.Linear(self.num_hidden_units)(x)) 
            x = hk.Linear(self.num_hidden_units)(x)
            x = self.activation(x + identity1)   

            identity2 = hk.Linear(self.num_hidden_units)(x)
            x = self.activation(identity2)
            x = self.activation(hk.Linear(self.num_hidden_units)(x))
            x = hk.Linear(self.num_hidden_units)(x) 
            x = self.activation(x + identity2)      

            identity3 = hk.Linear(self.num_hidden_units)(x) 
            x = self.activation(identity3)
            x = self.activation(hk.Linear(self.num_hidden_units)(x))    
            x = hk.Linear(self.num_hidden_units)(x) 
            x = self.activation(x + identity3)      

            V = hk.Linear(1)(x)[0]
            return V

class pi_function(hk.Module):
    def __init__(self, config, num_actions, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config["num_hidden_units"]
        self.activation = activation_dict[config["activation"]]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.num_actions = num_actions

    def __call__(self, obs):
        x = jnp.ravel(obs)

        identity1 = hk.Linear(self.num_hidden_units)(x) 
        x = self.activation(identity1)
        x = self.activation(hk.Linear(self.num_hidden_units)(x))
        x = hk.Linear(self.num_hidden_units)(x) 
        x = self.activation(x + identity1)     

        identity2 = hk.Linear(self.num_hidden_units)(x)
        x = self.activation(identity2)
        x = self.activation(hk.Linear(self.num_hidden_units)(x)) 
        x = hk.Linear(self.num_hidden_units)(x) 
        x = self.activation(x + identity2)     

        identity3 = hk.Linear(self.num_hidden_units)(x) 
        x = self.activation(identity3)
        x = self.activation(hk.Linear(self.num_hidden_units)(x)) 
        x = hk.Linear(self.num_hidden_units)(x) 
        x = self.activation(x + identity3)      

        pi_logit = hk.Linear(self.num_actions)(x)
        return pi_logit

def circuit_to_matrix(cnot_list, cardinality):
    matrix = jnp.eye(cardinality, dtype=jnp.float32)
    for control, target in cnot_list:
        if control == target:
            raise ValueError(f"Control and target cannot be the same: {control}")
        if control >= cardinality or target >= cardinality:
            raise ValueError(f"Control and target must be less than cardinality: {control}, {target} >= {cardinality}")
        matrix = matrix.at[target].set((matrix[target] + matrix[control]) % 2)
    return matrix

def load_model_custom(cardinality, topology):
    # 1. Trova la cartella dove risiede fisicamente questo file (scripts/)
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Il JSON è nella stessa cartella di questo script (scripts/)
    config_path = os.path.join(utils_dir, "config_custom_circuit.json")
    
    # 3. La cartella models è al piano di sopra rispetto a scripts/
    folder = os.path.join(os.path.dirname(utils_dir), "models")
    
    name_model = f"alphacnot_{cardinality}{topology}.params"
    path = os.path.join(folder, name_model)

    # Caricamento Params
    with open(path, 'rb') as f:
        params = pkl.load(f)

    # Caricamento JSON (ora il percorso è corretto!)
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Update cardinality and topology in config
    config["env_config"]["cardinality"] = cardinality
    config["env_config"]["topology"] = topology


    env = env.AlphaCNOT_env_topology(cardinality=cardinality, topology=topology)
    num_actions = env.num_actions()

    # Load the networks
    V_net = hk.without_apply_rng(hk.transform(lambda o: V_function(config)(o.astype(float))))
    pi_net = hk.without_apply_rng(hk.transform(lambda o: pi_function(config, num_actions)(o.astype(float))))

    return params, config, V_net, pi_net, env

def index_to_pair(idx, cardinality):
    i = idx // (cardinality - 1)
    j = idx % (cardinality - 1)
    j = j + (j >= i)
    return i, j

def generate_random_moves(key, cardinality, num_pairs):
        
        total = cardinality**2 - cardinality
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(
            subkey,
            total,
            shape=(num_pairs,),
            replace=True
        )
        i, j = jax.vmap(lambda x: index_to_pair(x, cardinality))(idx)

        return jnp.stack([i, j], axis=1)

def apply_xor(
        intial_matrix: chex.Array,
        action_index: chex.Array, 
        ) -> chex.Array:

        action_index = jnp.atleast_1d(action_index)
        control_idx, target_idx = action_index[0], action_index[1]
        control_row = intial_matrix[control_idx] 
        target_row = intial_matrix[target_idx]
        new_target_row = jnp.mod(target_row + control_row, 2)
        new_matrix = intial_matrix.at[target_idx].set(new_target_row)
        
        return new_matrix.astype(jnp.float32)

def get_recurrent_fn_simple(env, V_func, pi_func, config):

    def recurrent_fn(params, key, actions, env_states):

        env_states, obs, rewards, terminals, _ = jax.vmap(env.step)(actions, env_states)
        V = jax.vmap(V_func, in_axes=(None, 0))(params["V"], obs.astype(float))
        pi_logit = jax.vmap(pi_func, in_axes=(None, 0))(params["pi"], obs.astype(float))
        return mctx.RecurrentFnOutput(
            reward=rewards,
            discount=(1.0 - terminals) * config.get("discount", 1.0),
            prior_logits=jnp.reshape(pi_logit, (actions.shape[0], -1)), 
            value=jnp.reshape(V, (-1,)) 
        ), env_states
    
    return recurrent_fn

def apply_AlphaCNOT(model_bundle, cnot_list, card):
    """
    Return the optimized CNOT list after applying 
    the chosen AlphaCNOT model to the input CNOT list.
    """
    params, config, V_net, pi_net, env = model_bundle
    num_actions = env.num_actions()

    matrix = circuit_to_matrix(cnot_list, card)

    state = (matrix, jnp.array(0), jnp.array(0), jax.random.PRNGKey(0))
    
    rec_fn = get_recurrent_fn_simple(env, V_net.apply, pi_net.apply, config)
    
    @jax.jit
    def get_action(params, key, state):
        obs = env.get_observation(state)
        obs_batch = jnp.expand_dims(obs.astype(float), 0)
        state_batch = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), state)
        
        pi_logits = pi_net.apply(params["pi"], obs_batch)
        v_value = V_net.apply(params["V"], obs_batch)

        root = mctx.RootFnOutput(
            prior_logits=jnp.reshape(pi_logits, (1, num_actions)), 
            value=jnp.reshape(v_value, (1,)),
            embedding=state_batch
        )

        policy_output = mctx.gumbel_muzero_policy(
            params=params, rng_key=key, root=root, recurrent_fn=rec_fn,
            num_simulations=config.get("num_simulations", 16),
            max_num_considered_actions=num_actions,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=config.get("use_mixed_value", True),
                value_scale=config.get("value_scale", 1.0)
            ),
        )
        return policy_output.action[0]

    optimized_cnots = []
    max_steps = card**2
    master_key = jax.random.PRNGKey(42)
    
    for _ in range(max_steps):
        master_key, subkey = jax.random.split(master_key)
        action = get_action(params, subkey, state)
        state, _, _, done, info = env.step(action, state)
        control, target = index_to_pair(action, card)
        optimized_cnots.append((int(control), int(target)))
        
        if info.get("hamming_distance", 1.0) == 0 or done:
            break
            
    return optimized_cnots