"""Utilities for locating and reading AlphaCNOT model files."""

from collections import namedtuple
from pathlib import Path
import pickle


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_ScaleByAdamState = namedtuple("ScaleByAdamState", ("count", "mu", "nu"))
_EmptyState = namedtuple("EmptyState", ())


class _InferenceCheckpointUnpickler(pickle.Unpickler):
    """Load inference data even when Optax is not installed.

    Some full checkpoints contain Optax state objects even though inference only
    needs ``state.params``.  Replacing those two tuple types avoids making Optax
    a runtime dependency of the synthesis utilities.
    """

    _OPTAX_TYPES = {
        ("optax._src.transform", "ScaleByAdamState"): _ScaleByAdamState,
        ("optax._src.base", "EmptyState"): _EmptyState,
    }

    def find_class(self, module, name):
        replacement = self._OPTAX_TYPES.get((module, name))
        if replacement is not None:
            return replacement
        return super().find_class(module, name)


def normalize_terminal_mode(terminal_mode):
    """Return the canonical checkpoint terminal mode."""

    normalized = str(terminal_mode).strip().lower().replace("_", "-")
    if normalized in {"id", "identity", "identity-only"}:
        return "identity-only"
    if normalized in {
        "perm",
        "permutation",
        "permutation-invariant",
        "permutation-relaxed",
    }:
        return "permutation-invariant"
    raise ValueError(
        "terminal_mode must be 'identity' or 'permutation-relaxed' "
        "(aliases: 'id', 'perm', 'identity-only', 'permutation-invariant')."
    )


def normalize_reward_type(reward_type):
    """Return the short reward label used in pretrained directory names."""

    normalized = str(reward_type).strip().lower()
    if normalized in {"inf", "informed"}:
        return "inf"
    if normalized in {"mix", "mixed"}:
        return "mix"
    raise ValueError("reward_type must be 'inf'/'informed' or 'mix'/'mixed'.")


def resolve_trained_model_path(
    cardinality,
    topology,
    reward_type="mix",
    terminal_mode="identity",
    repository_root=REPOSITORY_ROOT,
):
    """Resolve one of the packaged ``.pkl`` inference checkpoints."""

    cardinality = int(cardinality)
    topology = "ALL" if topology is None else str(topology).upper()
    if cardinality == 8 and topology == "T":
        topology = "T1"

    terminal_mode = normalize_terminal_mode(terminal_mode)
    model_suffix = "perm" if terminal_mode == "permutation-invariant" else "id"

    if topology == "ALL":
        reward_type = normalize_reward_type(reward_type)
        family = "perm" if model_suffix == "perm" else "identity"
        model_group = f"{family}-{reward_type}"
    else:
        model_group = f"top-{model_suffix}"

    model_name = f"alphacnot_{cardinality}{topology}_{model_suffix}.pkl"
    model_path = (
        Path(repository_root)
        / "trained_models"
        / f"Card_{cardinality}"
        / model_group
        / model_name
    )

    if not model_path.is_file():
        card_directory = Path(repository_root) / "trained_models" / f"Card_{cardinality}"
        available = sorted(path.relative_to(card_directory).as_posix() for path in card_directory.glob("*/*.pkl"))
        available_text = "\n  - ".join(available) if available else "none"
        raise FileNotFoundError(
            f"AlphaCNOT checkpoint not found: {model_path}\n"
            f"Available models for cardinality {cardinality}:\n  - {available_text}"
        )

    return model_path


def load_model_file(model_path):
    """Load parameters and optional metadata from a ``.pkl`` or ``.params`` file."""

    model_path = Path(model_path)
    with model_path.open("rb") as model_file:
        payload = _InferenceCheckpointUnpickler(model_file).load()

    checkpoint_config = None
    checkpoint_metadata = {}

    if isinstance(payload, dict) and "state" in payload:
        state = payload.get("state")
        if not isinstance(state, dict) or "params" not in state:
            raise ValueError(f"Checkpoint has no state.params entry: {model_path}")
        params = state["params"]
        checkpoint_config = payload.get("config")
        checkpoint_metadata = {
            "schema": payload.get("schema"),
            "terminal_mode": payload.get("terminal_mode"),
            "training_protocol": payload.get("training_protocol"),
            "informed_phase": state.get("informed_phase"),
            "update": state.get("update"),
        }
    else:
        # Backward compatibility for locally trained legacy .params files.
        params = payload

    if not isinstance(params, dict) or not {"V", "pi"}.issubset(params):
        raise ValueError(f"Invalid AlphaCNOT parameters in {model_path}")

    return _namespace_haiku_params(params), checkpoint_config, checkpoint_metadata


def runtime_config_from_checkpoint(
    checkpoint_config,
    checkpoint_metadata,
    cardinality,
    topology,
    requested_terminal_mode,
):
    """Translate checkpoint field names to the repository's inference API."""

    terminal_mode = checkpoint_metadata.get("terminal_mode")
    if terminal_mode is None:
        terminal_mode = checkpoint_config.get("terminal_mode")
    if terminal_mode is None:
        terminal_mode = checkpoint_config.get("terminal", requested_terminal_mode)
    terminal_mode = normalize_terminal_mode(terminal_mode)

    requested_terminal_mode = normalize_terminal_mode(requested_terminal_mode)
    if terminal_mode != requested_terminal_mode:
        raise ValueError(
            f"Requested terminal mode {requested_terminal_mode!r}, but the "
            f"checkpoint uses {terminal_mode!r}."
        )

    reward = str(checkpoint_config.get("reward", "informed")).lower()
    if checkpoint_config.get("switch_success_threshold", 0):
        informed_reward = bool(checkpoint_metadata.get("informed_phase", True))
    else:
        update = max(0, int(checkpoint_metadata.get("update") or 0) - 1)
        switch_update = int(
            checkpoint_config.get("steps", 0)
            * checkpoint_config.get("switch_fraction", 0.5)
        )
        informed_reward = reward == "informed" or (
            reward == "mixed" and update < switch_update
        )

    return {
        "num_hidden_layers": checkpoint_config.get("layers", 9),
        "num_hidden_units": checkpoint_config.get("width", 256),
        "discount": checkpoint_config.get("discount", 0.99),
        "use_mixed_value": checkpoint_config.get("use_mixed_value", True),
        "value_scale": checkpoint_config.get("value_scale", 0.1),
        "value_target": checkpoint_config.get("value_target", "maxq"),
        "target_update_frequency": checkpoint_config.get("target_update", 100),
        "num_simulations": checkpoint_config.get("simulations", 50),
        "max_depth": checkpoint_config.get("max_depth"),
        "max_steps": checkpoint_config.get("max_steps", 4 * int(cardinality) ** 2),
        "gumbel_scale": checkpoint_config.get("gumbel_scale", 1.0),
        "maxvisit_init": checkpoint_config.get("maxvisit_init", 50.0),
        "q_rescale": checkpoint_config.get("q_rescale", True),
        "q_epsilon": checkpoint_config.get("q_epsilon", 1e-8),
        "environment": "env_AlphaCNOT",
        "env_config": {
            "cardinality": int(cardinality),
            "topology": topology,
            "informed_reward": informed_reward,
            "terminal_mode": terminal_mode,
        },
        "activation": checkpoint_config.get("activation", "elu"),
    }


def _namespace_haiku_params(params):
    """Add the module prefixes expected by the repository's Haiku modules."""

    namespaced = {}
    for network_name, module_name in (("V", "v_function"), ("pi", "pi_function")):
        network_params = params[network_name]
        if not isinstance(network_params, dict):
            namespaced[network_name] = network_params
            continue

        namespaced[network_name] = {
            key if "/" in key else f"{module_name}/{key}": value
            for key, value in network_params.items()
        }
    return namespaced
