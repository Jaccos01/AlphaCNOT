import jax as jx
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import chex


def index_to_pair(idx, cardinality):

    i = idx // (cardinality - 1)
    j = idx % (cardinality - 1)
    j = j + (j >= i)

    return i, j

def index_to_pair_topology(idx, topology, cardinality):

    i, j = topology[idx]

    return i, j

# 1. Fully Connected
def fully_connected_topology(cardinality):

    total_actions = cardinality * (cardinality - 1)
    topology = jnp.zeros((total_actions, 2), dtype=jnp.int32)
    index = 0
    for i in range(cardinality):
        for j in range(cardinality):
            if i != j:
                topology = topology.at[index].set(jnp.array([i, j], dtype=jnp.int32))
                index += 1

    return topology

# 2. Linear
def linear_topology(cardinality):

    total_actions = 2 * (cardinality - 1)
    topology = jnp.zeros((total_actions, 2), dtype=jnp.int32)
    index = 0
    for i in range(cardinality - 1):
        topology = topology.at[index].set(jnp.array([i, i + 1], dtype=jnp.int32))
        index += 1
        topology = topology.at[index].set(jnp.array([i + 1, i], dtype=jnp.int32))
        index += 1

    return topology

# 3. T-Topology
def t_topology(cardinality):

    total_actions = 2* (cardinality - 1)
    topology = jnp.zeros((total_actions, 2), dtype=jnp.int32)

    topology = topology.at[0].set(jnp.array([0, 1], dtype=jnp.int32))
    topology = topology.at[1].set(jnp.array([1, 0], dtype=jnp.int32))
    topology = topology.at[2].set(jnp.array([1, 2], dtype=jnp.int32))
    topology = topology.at[3].set(jnp.array([2, 1], dtype=jnp.int32))
    topology = topology.at[4].set(jnp.array([1, 3], dtype=jnp.int32))
    topology = topology.at[5].set(jnp.array([3, 1], dtype=jnp.int32))

    index = 6
    for i in range(3, cardinality-1):
        topology = topology.at[index].set(jnp.array([i, i+1], dtype=jnp.int32))
        index += 1
        topology = topology.at[index].set(jnp.array([i+1, i], dtype=jnp.int32))
        index += 1

    return topology

# 4. H-Topology
def h_topology(cardinality):

    total_actions = 2* (cardinality - 1)
    topology = jnp.zeros((total_actions, 2), dtype=jnp.int32)

    topology = topology.at[0].set(jnp.array([0, 1], dtype=jnp.int32))
    topology = topology.at[1].set(jnp.array([1, 0], dtype=jnp.int32))
    topology = topology.at[2].set(jnp.array([1, 2], dtype=jnp.int32))
    topology = topology.at[3].set(jnp.array([2, 1], dtype=jnp.int32))
    topology = topology.at[4].set(jnp.array([1, 3], dtype=jnp.int32))
    topology = topology.at[5].set(jnp.array([3, 1], dtype=jnp.int32))

    index = 6
    for i in range(3, cardinality-3):
        topology = topology.at[index].set(jnp.array([i, i+1], dtype=jnp.int32))
        index += 1
        topology = topology.at[index].set(jnp.array([i+1, i], dtype=jnp.int32))
        index += 1

    i = cardinality-3
    topology = topology.at[index].set(jnp.array([i,i+1], dtype=jnp.int32))
    topology = topology.at[index+1].set(jnp.array([i+1, i], dtype=jnp.int32))
    topology = topology.at[index+2].set(jnp.array([i, i+2], dtype=jnp.int32))
    topology = topology.at[index+3].set(jnp.array([i+2, i], dtype=jnp.int32))

    return topology

# 5. Y-Topology
def y_topology(cardinality):

    total_actions = 2* (cardinality - 1)
    topology = jnp.zeros((total_actions, 2), dtype=jnp.int32)

    if cardinality % 2 == 0:
        head = int(cardinality/2)
    else:
        head = int((cardinality+1)/2)
    index = 0

    for i in range(0,head):
        topology = topology.at[index].set(jnp.array([i, i+1], dtype=jnp.int32))
        index += 1
        topology = topology.at[index].set(jnp.array([i+1, i], dtype=jnp.int32))
        index += 1

    if cardinality % 2 == 0:
        topology = topology.at[index].set(jnp.array([head-1, head+1], dtype=jnp.int32))
        index += 1
        topology = topology.at[index].set(jnp.array([head+1, head-1], dtype=jnp.int32))
        index += 1
    else:
        topology = topology.at[index].set(jnp.array([head-2, head+1], dtype=jnp.int32))
        index += 1
        topology = topology.at[index].set(jnp.array([head+1, head-2], dtype=jnp.int32))
        index += 1
    for j in range(head+1,cardinality-1):
        topology = topology.at[index].set(jnp.array([j, j+1], dtype=jnp.int32))
        index += 1
        topology = topology.at[index].set(jnp.array([j+1, j], dtype=jnp.int32))
        index += 1

    return topology

# 6. F-topology
def f_topology(cardinality):

    total_actions = 2 * (cardinality - 1)
    topology = jnp.zeros((total_actions, 2), dtype=jnp.int32)
    index = 0
    for i in range(cardinality - 2):
        topology = topology.at[index].set(jnp.array([i, i + 1], dtype=jnp.int32))
        index += 1
        topology = topology.at[index].set(jnp.array([i + 1, i], dtype=jnp.int32))
        index += 1
    topology = topology.at[index].set(jnp.array([2, cardinality - 1], dtype=jnp.int32))
    index += 1
    topology = topology.at[index].set(jnp.array([cardinality - 1, 2], dtype=jnp.int32))

    return topology


def topology_8h():

    topology = jnp.array([
    [0, 2], [2, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3],
    [4, 5], [5, 4], [4, 6], [6, 4], [6, 7], [7, 6]
    ], dtype=jnp.int32)

    return topology

def topology_8t2():

    topology = jnp.array([
    [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3],
    [4, 5], [5, 4], [3, 6], [6, 3], [6, 7], [7, 6]
    ], dtype=jnp.int32)

    return topology

def topology_9h1():

    topology = jnp.array([
    [0, 2], [2, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 5], [5, 3],
    [5, 6], [6, 5], [7, 6], [6, 7], [4, 7], [7, 4], [8, 7], [7, 8]
    ], dtype=jnp.int32)

    return topology

def topology_8lswap():

    topology= jnp.array([
    [0, 2], [2, 0], [2, 1], [1, 2], [1, 3], [3, 1], [3, 4], [4, 3],
    [4, 5], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6]
    ], dtype=jnp.int32)

    return topology

class env_AlphaCNOT:
    def __init__(
        self,
        cardinality,
        topology=None,
        informed_reward=True,
        terminal_mode="identity-only",
    ):

        self.cardinality = cardinality
        self.identity = jnp.identity(cardinality, dtype=jnp.float32)
        self.informed_reward = informed_reward
        terminal_mode = str(terminal_mode).lower().replace("_", "-")
        if terminal_mode in {"identity", "id"}:
            terminal_mode = "identity-only"
        elif terminal_mode in {"permutation", "perm", "permutation-relaxed"}:
            terminal_mode = "permutation-invariant"
        if terminal_mode not in {"identity-only", "permutation-invariant"}:
            raise ValueError(f"Unknown terminal mode: {terminal_mode}")
        self.terminal_mode = terminal_mode

        # Initial operations ---> n^2
        self.max_operations = jnp.floor((self.cardinality**2)).astype(jnp.int32)
        self.limit = jnp.floor((self.cardinality**2)/jnp.log2(self.cardinality)).astype(jnp.int32)

        if topology is None or topology == "ALL":
            self.topology = fully_connected_topology(cardinality)
        elif topology == "L" and cardinality != 8:
            self.topology = linear_topology(cardinality)
        elif topology == "T" or topology == "T1":
            self.topology = t_topology(cardinality)
        elif topology == "Y":
            self.topology = y_topology(cardinality)
        elif topology == "F":
            self.topology = f_topology(cardinality)
        elif topology == "H" and cardinality != 8:
            self.topology = h_topology(cardinality)
        elif topology == "H" and cardinality == 8:
            self.topology = topology_8h()
        elif topology == "T2" and cardinality==8:
            self.topology = topology_8t2()
        elif topology == "L-SWAP":
            self.topology = topology_8lswap()
        else:
            raise ValueError(
                f"Topology {topology!r} is not available for cardinality {cardinality}."
            )

        self._num_actions = self.topology.shape[0]

    def generate_random_moves_topological(self, key, cardinality, num_pairs):

        total = self._num_actions
        key, subkey = jx.random.split(key)
        idx = jx.random.choice(
            subkey,
            total,
            shape=(num_pairs,),
            replace=True
        )
        i, j = jx.vmap(lambda x: index_to_pair_topology(x, self.topology, self.cardinality))(idx)

        return jnp.stack([i, j], axis=1)

    def generate_random_moves(self, key, cardinality, num_pairs):

        total = cardinality * (cardinality - 1)

        key, subkey = jx.random.split(key)
        idx = jx.random.choice(
            subkey,
            total,
            shape=(num_pairs,),
            replace=True
        )
        i, j = jx.vmap(lambda x: index_to_pair(x, cardinality))(idx)

        return jnp.stack([i, j], axis=1)


    def apply_xor(
        self,
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

    @partial(jit, static_argnums=(0,))
    def goal_distance(self, matrix):
        """Distance from the exact or permutation-relaxed terminal set."""

        matrix = matrix.astype(jnp.float32)
        if self.terminal_mode == "permutation-invariant":
            column_sums = jnp.sum(matrix, axis=0, dtype=jnp.int32)
            return jnp.sum(jnp.maximum(column_sums - 1, 0), dtype=jnp.int32)
        return jnp.sum(
            jnp.bitwise_xor(matrix.astype(jnp.int32), self.identity.astype(jnp.int32))
        ).astype(jnp.float32)

    @partial(jit, static_argnums=(0,))
    def is_goal(self, matrix):
        if self.terminal_mode == "permutation-invariant":
            return jnp.all(jnp.sum(matrix, axis=0) == 1) & jnp.all(
                jnp.sum(matrix, axis=1) == 1
            )
        return self.goal_distance(matrix) <= 0.001

    @partial(jit, static_argnums=(0,))
    def step(self, action, env_state):

        matrix, operation, init_op, key = env_state
        operation += 1
        terminal = False
        pair = index_to_pair_topology(action, self.topology, self.cardinality)

        action = jnp.array(pair)

        final_matrix = self.apply_xor(matrix, action)
        goal_distance = self.goal_distance(final_matrix)

        hamming_reward = - (goal_distance / self.cardinality)
        reward = jnp.where(self.informed_reward, hamming_reward, -1)

        env_state = (final_matrix, operation, init_op, key)

        terminal_1 = self.is_goal(final_matrix)
        reward = jnp.where(terminal_1, 1, reward)

        terminal_2 = jnp.where(operation >= (4*(self.cardinality**2)), True, False)

        terminal=jnp.logical_or(terminal_1,terminal_2)

        return env_state, self.get_observation(env_state), reward, terminal, {
            "hamming_distance": goal_distance,
            "goal_distance": goal_distance,
            "is_solved": terminal_1,
        }


    @partial(jit, static_argnums=(0,))
    def reset(self,key):

        operation = 0
        key, subkey = jx.random.split(key)
        init_operations = int(self.max_operations)
        moves = self.generate_random_moves(subkey, self.cardinality, init_operations)

        matrix = self.identity
        matrix, _ = jx.lax.scan(lambda carry, action: (self.apply_xor(carry, action), None),
                        matrix, moves)
        matrix = jnp.round(matrix)
        matrix = jnp.clip(matrix, 0.0, 1.0)
        matrix = matrix.astype(jnp.float32)

        env_state = (matrix, operation, init_operations, key)

        return env_state


    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state) -> chex.Array:

        matrix, _, _, _ = env_state
        obs = jnp.stack([matrix.astype(jnp.float32).ravel()], axis=-1)

        return obs

    def num_actions(self):
        return self._num_actions
