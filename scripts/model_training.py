import os
import json
from tqdm import tqdm
import pickle as pkl
import functools
from optimizers import adamw
import random

import jax as jx
from jax import jit, vmap, grad, value_and_grad
from jax import numpy as jnp
import mctx
import haiku as hk

from utils import V_function, pi_function
from utils import parse_train_args, dotdict
import env




def model_training(model_name, n_qubits, topology, reward, config, epochs, output_dir):

    if isinstance(config, dict):
        config = dotdict(config)
        
    Card = n_qubits
    Topology = topology


    print(f"Running training for cardinality: {Card}")
    print(f"Topology: {Topology}")

    if reward == "mix":
        first_stage = int((epochs/2))
        second_stage = epochs - first_stage
    else:
        first_stage = epochs
        second_stage = 0

    # Random seed
    seed = jx.random.PRNGKey(random.randint(0, 10000))

    config.update({"agent_type":"basic_tree_search", "seed":seed})
    config.update({"num_epochs": epochs})
    config.update({"env_config":{"cardinality":Card, "topology":Topology}})

    # Environment setup
    Environment = getattr(env, config.environment)
    # Configuration setup
    env_config = config.env_config

    folder_path = output_dir
    os.makedirs(folder_path, exist_ok=True)

    json_config_path = os.path.join(folder_path, f"{model_name}.json")
    out_path = os.path.join(folder_path, f"{model_name}.out")
    params_path = os.path.join(folder_path, f"{model_name}.params")

    if reward == "mix":
        out_path_first_stage = os.path.join(folder_path, f"{model_name}_first_stage.out")
        params_path_first_stage = os.path.join(folder_path, f"{model_name}_first_stage.params")


    data_to_save = {
        "training_information": config
    }

    with open(json_config_path, "w") as f:
        json.dump(data_to_save, f, indent=4)



    def get_recurrent_fn(env, V_func, pi_func):
        batch_step = vmap(env.step, in_axes=(0,0))
        batch_pi_func = vmap(pi_func,in_axes=(None,0))
        batch_V_func = vmap(V_func,in_axes=(None,0))
        def recurrent_fn(params, key, actions, env_states):
            V_params = params["V"]
            pi_params = params["pi"]
            key, subkey = jx.random.split(key)
            subkeys = jx.random.split(subkey, num=config.batch_size)

            env_states, obs, rewards, terminals, _ = batch_step(actions, env_states)

            
            V = batch_V_func(V_params, obs.astype(float))
            pi_logit = batch_pi_func(pi_params, obs.astype(float))
            recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=rewards,
            discount=(1.0-terminals)*config.discount,
            prior_logits=pi_logit,
            value=V)
            return recurrent_fn_output, env_states
        return recurrent_fn

    def get_init_fn(env):
        batch_reset = vmap(env.reset)

        def init_fn(key):
            dummy_state = env.reset(key)
            obs = env.get_observation(dummy_state)
            dummy_obs = obs.astype(float)

            key, subkey = jx.random.split(key)
            subkeys = jx.random.split(subkey, num=config.batch_size)

            env_states = batch_reset(subkeys)

            num_actions = env.num_actions()

            V_net = hk.without_apply_rng(hk.transform(lambda obs: V_function(config)(obs.astype(float))))
            key, subkey = jx.random.split(key)
            V_params = V_net.init(subkey, dummy_obs)
            V_func = V_net.apply

            V_target_params = V_params

            pi_net = hk.without_apply_rng(hk.transform(lambda obs: pi_function(config, num_actions)(obs.astype(float))))
            key, subkey = jx.random.split(key)
            pi_params = pi_net.init(subkey, dummy_obs)
            pi_func = pi_net.apply

            V_opt_init, V_opt_update, get_V_params = adamw(config.V_alpha, eps=config.eps_adam, b1=config.b1_adam, b2=config.b2_adam, wd=config.wd_adam)
            V_opt_state = V_opt_init(V_params)

            pi_opt_init, pi_opt_update, get_pi_params = adamw(config.pi_alpha, eps=config.eps_adam, b1=config.b1_adam, b2=config.b2_adam, wd=config.wd_adam)
            pi_opt_state = pi_opt_init(pi_params)

            return (env_states, V_func, pi_func, V_opt_state, pi_opt_state, 
                V_opt_update, pi_opt_update, get_V_params, V_target_params, 
                get_pi_params, V_opt_init, pi_opt_init)
        return init_fn

    def get_AC_loss(pi_func, V_func):
        def AC_loss(pi_params, V_params, pi_target, V_target, obs):
            pi_logits = pi_func(pi_params, obs.astype(float))
            V = V_func(V_params, obs.astype(float))

            pi_loss = jnp.sum(pi_target*(jnp.log(pi_target)-jx.nn.log_softmax(pi_logits)))
            V_loss = (V_target-V)**2

            return jnp.sum(pi_loss+V_loss)
        return AC_loss

    def get_agent_environment_interaction_loop_function(env, V_func, pi_func, recurrent_fn, V_opt_update, pi_opt_update, get_V_params, get_pi_params, num_actions, iterations):
        batch_loss = lambda *x: jnp.mean(vmap(get_AC_loss(pi_func, V_func), in_axes=(None,None,0,0,0))(*x))
        loss_grad = grad(batch_loss, argnums=(0,1))
        loss_value_and_grad = value_and_grad(batch_loss, argnums=(0,1))
        batch_step = vmap(env.step, in_axes=(0,0))
        batch_obs = vmap(env.get_observation)
        batch_reset = vmap(env.reset)
        batch_V_func = vmap(V_func,in_axes=(None,0))
        batch_pi_func = vmap(pi_func,in_axes=(None,0))

        def agent_environment_interaction_loop_function(S):
            def loop_function(S, data):
                obs = batch_obs(S["env_states"])
                pi_logits = batch_pi_func(get_pi_params(S["pi_opt_state"]), obs.astype(float))
                V = batch_V_func(S["V_target_params"], obs.astype(jnp.float32))

                root = mctx.RootFnOutput(
                prior_logits=pi_logits,
                value=V,
                embedding=S["env_states"]
                )

                S["key"], subkey = jx.random.split(S["key"])
                policy_output = mctx.gumbel_muzero_policy(
                params={"V":S["V_target_params"], "pi":get_pi_params(S["pi_opt_state"])},
                rng_key=subkey,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=config.num_simulations,
                max_num_considered_actions=num_actions,
                qtransform=functools.partial(
                    mctx.qtransform_completed_by_mix_value,
                    use_mixed_value=config.use_mixed_value,
                    value_scale=config.value_scale
                ),
                )


                search_policy = policy_output.action_weights
                if(config.value_target=='maxq'):
                    search_value = policy_output.search_tree.qvalues(jnp.full(config.batch_size, policy_output.search_tree.ROOT_INDEX))[jnp.arange(config.batch_size), policy_output.action]
                elif(config.value_target=='nodev'):
                    search_value = policy_output.search_tree.node_values[:, policy_output.search_tree.ROOT_INDEX]
                else:
                    raise ValueError("Unknown value target.")

                loss_val, (pi_grads, V_grads) = loss_value_and_grad(
                    get_pi_params(S["pi_opt_state"]), 
                    get_V_params(S["V_opt_state"]), 
                    search_policy, 
                    search_value, 
                    obs
                )
                
                S["pi_opt_state"] = pi_opt_update(S["opt_t"], pi_grads, S["pi_opt_state"])
                S["V_opt_state"] = V_opt_update(S["opt_t"], V_grads, S["V_opt_state"])
                S["opt_t"]+=1

                # Update target params after a particular number of parameter updates
                S["V_target_params"] = jx.tree_util.tree_map(lambda x,y: jnp.where(S["opt_t"]%config.target_update_frequency==0,x,y),get_V_params(S["V_opt_state"]), S["V_target_params"])

                # always take action recommended by tree search
                actions = policy_output.action

                S["env_states"], obs, reward, terminal, info = batch_step(actions, S["env_states"])

                # reset environment if terminated
                S["key"], subkey = jx.random.split(S["key"])
                subkeys = jx.random.split(subkey, num=config.batch_size)
                S["env_states"] = jx.tree_util.tree_map(lambda x,y: jnp.where(jnp.reshape(terminal,[terminal.shape[0]]+[1]*(len(x.shape)-1)), x,y), batch_reset(subkeys), S["env_states"])

                # update statistics for computing average return
                S["episode_return"] += reward
                S["avg_return"] = jnp.where(terminal, S["avg_return"]*config.avg_return_smoothing+S["episode_return"]*(1-config.avg_return_smoothing), S["avg_return"])
                S["episode_return"] = jnp.where(terminal, 0, S["episode_return"])
                S["num_episodes"] = jnp.where(terminal, S["num_episodes"]+1, S["num_episodes"])

                # Metrics logging
                current_metrics = {
                    "num_episodes": S["num_episodes"],
                    "reward_current_step": reward,
                    "hamming_distance": info["hamming_distance"],
                }
                return S, current_metrics

            S["key"], subkey = jx.random.split(S["key"])
            S, history = jx.lax.scan(loop_function, S, None, length=iterations)

            return S, history
        return agent_environment_interaction_loop_function


    opt_t = 0
    time_step = 0
    avg_return = jnp.zeros(config.batch_size)
    episode_return = jnp.zeros(config.batch_size)
    num_episodes = jnp.zeros(config.batch_size)

    env = Environment(**env_config)
    num_actions = env.num_actions()

    key, subkey = jx.random.split(key)

    (env_states, V_func, pi_func, V_opt_state, pi_opt_state, 
    V_opt_update, pi_opt_update, get_V_params, V_target_params, 
    get_pi_params, V_opt_init, pi_opt_init) = get_init_fn(env)(subkey)



    recurrent_fn = get_recurrent_fn(env, V_func, pi_func)

    agent_environment_interaction_loop_function = jit(get_agent_environment_interaction_loop_function(env, V_func, pi_func, 
                                                        recurrent_fn, V_opt_update, pi_opt_update, get_V_params, get_pi_params, 
                                                        num_actions, config.eval_frequency))
    run_state_names = ["env_states", "V_opt_state", "V_target_params", "pi_opt_state", "opt_t", "avg_return", "episode_return", "num_episodes", "key"]
    var_dict = locals()
    run_state = {name:var_dict[name] for name in run_state_names}

    avg_returns = []
    times = []




    # --- PHASE 1: INFORMED REWARD ---
    print("🚀 Starting PHASE 1: Informed Reward")

    for i in tqdm(range(first_stage // config.eval_frequency)):
        run_state, _ = agent_environment_interaction_loop_function(run_state)


        valid_avg_returns = run_state["avg_return"][run_state["num_episodes"] > 0]
        valid_num_episodes = run_state["num_episodes"][run_state["num_episodes"] > 0]
        
        if len(valid_avg_returns) > 0:
            current_avg_ret = jnp.mean(valid_avg_returns / (1 - config.avg_return_smoothing**valid_num_episodes))
            print(f"Running Average Return: {current_avg_ret:.4f}")
            avg_returns.append(float(current_avg_ret))
        else:
            print("Running Average Return: N/A (No episodes completed yet)")
            avg_returns.append(0.0)

        time_step += config.eval_frequency
        times.append(time_step)




    if reward == "mix":
        # Saving of the model from the first phase (informed reward) before starting the second phase
        with open(out_path_first_stage, 'wb') as f:
            pkl.dump({
                'config': dict(config),
                'avg_returns': avg_returns,
                'times': times
            }, f)
        with open(params_path_first_stage, 'wb') as f:
            pkl.dump({
                'V' : get_V_params(run_state['V_opt_state']),
                'pi' : get_pi_params(run_state['pi_opt_state'])
            }, f)

        print("\n🔄 Switching to NOT INFORMED REWARD...")

        env_non_informed = Environment(cardinality=Card, topology=Topology, informed_reward=False)

        recurrent_fn_2 = get_recurrent_fn(env_non_informed, V_func, pi_func)
        agent_env_loop_non_informed = jit(get_agent_environment_interaction_loop_function(
            env_non_informed, V_func, pi_func, recurrent_fn_2, 
            V_opt_update, pi_opt_update, get_V_params, get_pi_params, 
            num_actions, config.eval_frequency
        ))

        # --- PHASE 2: NOT INFORMED REWARD  ---
        print("🚀 Starting PHASE 2: Not Informed Reward")
        for i in tqdm(range(second_stage // config.eval_frequency)):
            run_state, _ = agent_env_loop_non_informed(run_state)


            valid_avg_returns = run_state["avg_return"][run_state["num_episodes"] > 0]
            valid_num_episodes = run_state["num_episodes"][run_state["num_episodes"] > 0]

            
            if len(valid_avg_returns) > 0:
                current_avg_ret = jnp.mean(valid_avg_returns / (1 - config.avg_return_smoothing**valid_num_episodes))
                print(f"Running Average Return: {current_avg_ret:.4f}")
                avg_returns.append(float(current_avg_ret))
            else:
                print("Running Average Return: N/A (No episodes completed yet)")
                avg_returns.append(0.0)

            time_step += config.eval_frequency
            times.append(time_step)


    with open(out_path, 'wb') as f:
        pkl.dump({
            'config': dict(config),
            'avg_returns': avg_returns,
            'times': times
        }, f)

    with open(params_path, 'wb') as f:
        pkl.dump({
            'V' : get_V_params(run_state['V_opt_state']),
            'pi' : get_pi_params(run_state['pi_opt_state'])
        }, f)

    print(f"n✅ Task completed successfully!")
    print(f"📁 All files saved in: {folder_path}")
