import jax
import jax.numpy as jnp
import haiku as hk
import pickle as pkl
import json
from tqdm import tqdm
import numpy as np
import mctx
import functools
import os
import time

from scripts.utils import V_function, pi_function, get_recurrent_fn_simple, env_CNOT

def run_validation(config, params_path, model_folder, n_test, n_rep, top):
    # Parameters loading
    with open(params_path, 'rb') as f:
        params = pkl.load(f)

    Environment = getattr(env_CNOT, config["environment"])
    env = Environment(**config["env_config"])
    num_actions = env.num_actions()
    card = config['env_config']['cardinality']

    # Setup
    V_net = hk.without_apply_rng(hk.transform(lambda o: V_function(config)(o.astype(float))))
    pi_net = hk.without_apply_rng(hk.transform(lambda o: pi_function(config, num_actions)(o.astype(float))))

    rec_fn = get_recurrent_fn_simple(env, V_net.apply, pi_net.apply, config)

    @functools.partial(jax.jit, static_argnums=(3, 4, 5, 6))
    def get_action(params, key, state, env, pi_net, V_net, recurrent_fn):
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
            params=params,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config["num_simulations"],
            max_num_considered_actions=num_actions,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=config.get("use_mixed_value", True),
                value_scale=config.get("value_scale", 0.1)
            ),
        )
        return policy_output.action[0]

    model_dir_name = os.path.basename(os.path.normpath(model_folder))
    
    # Saving results
    validation_root = "validation_results"
    specific_val_dir = os.path.join(validation_root, f"Validation_{model_dir_name}")
    os.makedirs(specific_val_dir, exist_ok=True)

    
    val_out_dir = os.path.join(model_folder, "out")
    os.makedirs(val_out_dir, exist_ok=True)

    results = []
    success_count = 0
    all_episodes_moves = []
    master_key = jax.random.PRNGKey(int(time.time()))
    max_steps = 2 * (card**2)

    print(f"🚀 Validation: {card}Q | Top: {top} | {n_test} test matrices")

    start_val_time = time.time()

    for i in tqdm(range(n_test)):
        test_key = jax.random.PRNGKey(i + 777)
        results_for_i = []
        moves_for_i = []

        for rep_idx in range(n_rep):
            state = env.reset(test_key)
            done = False
            steps = 0
            current_episode_actions = []

            while not done and steps < max_steps:
                master_key, search_key = jax.random.split(master_key)
                action = get_action(params, search_key, state, env, pi_net, V_net, rec_fn)
                
                current_episode_actions.append(int(action)) 
                state, obs, reward, terminal, info = env.step(action, state)
                steps += 1
                
                if info.get("hamming_distance", 1.0) == 0:
                    success_count += 1
                    results_for_i.append(steps)
                    done = True
            
            if not done:
                results_for_i.append(max_steps + 1)
            
            moves_for_i.append({
                "rep": rep_idx,
                "solved": bool(done),
                "steps": int(steps),
                "actions": current_episode_actions
            })
        
        # Best results of the repetitions for the current test matrix
        results.append(min(results_for_i))
        all_episodes_moves.append({"test_id": i, "repetitions": moves_for_i})

    duration = time.time() - start_val_time
    successful_results = [r for r in results if r <= max_steps]
    
    # --- STATS ---
    stats = {
        "success_rate_pct": float(len(successful_results) / n_test * 100),
        "avg_steps_success": float(np.mean(successful_results)) if successful_results else 0.0,
        "duration_sec": float(duration),
        "n_test": n_test,
        "n_rep": n_rep
    }

    prefix = f"val_{card}Q_{top}_rep{n_rep}"
    stats_path = os.path.join(specific_val_dir, f"{prefix}_stats.json")
    moves_path = os.path.join(specific_val_dir, f"{prefix}_actions.json")

    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    with open(moves_path, 'w') as f:
        json.dump(all_episodes_moves, f, indent=2)

    print(f"✅ Risultati salvati in: {specific_val_dir}")