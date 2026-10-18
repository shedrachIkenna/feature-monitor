"""Load the active profile's language model and its residual-stream SAE.

A profile (monitor/profiles.yaml) names the model, its SAE, the layer the SAE reads and the
matching Neuronpedia labels. FEATURE_MONITOR_PROFILE picks one, default gpt2_small. Other modules
ask this one for the profile instead of naming a model themselves.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import yaml
from sae_lens import SAE
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoConfig

PROFILES_PATH = Path(__file__).resolve().with_name("profiles.yaml")
PROFILE_ENV = "FEATURE_MONITOR_PROFILE"
DEFAULT_PROFILE = "gpt2_small"
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class Profile:
    """One model, the SAE that reads it and where that SAE's feature labels live."""

    name: str
    label: str
    model_name: str
    dtype: str
    process_weights: bool
    sae_release: str
    sae_id: str
    layer: int
    neuronpedia_id: str

    @property
    def file_stem(self) -> str:
        """Prefix of this profile's baseline, threshold and cache files, e.g. gpt2_8."""
        return f"{self.model_name.split('/')[-1]}_{self.layer}"


def load_profiles(path: Path = PROFILES_PATH) -> dict[str, Profile]:
    """Read every profile; unknown or missing keys raise TypeError, a bad dtype ValueError."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    profiles = {name: Profile(name=name, **fields) for name, fields in raw.items()}
    for profile in profiles.values():
        if profile.dtype not in DTYPES:
            raise ValueError(f"profile {profile.name}: dtype must be one of {sorted(DTYPES)}")
    return profiles


def active_profile() -> Profile:
    """Return the profile named by FEATURE_MONITOR_PROFILE, or gpt2_small when it is unset."""
    name = os.environ.get(PROFILE_ENV, DEFAULT_PROFILE)
    profiles = load_profiles()
    if name not in profiles:
        raise ValueError(f"unknown {PROFILE_ENV}={name!r}; choose one of {sorted(profiles)}")
    return profiles[name]


def truncation(profile: Profile) -> dict[str, int]:
    """Config override keeping only the blocks up to the SAE's layer (later ones cannot affect it)."""
    config = AutoConfig.from_pretrained(profile.model_name)
    # TransformerLens writes overrides straight into the config, past its name aliases.
    key = config.attribute_map.get("num_hidden_layers", "num_hidden_layers")
    return {key: profile.layer + 1}


@lru_cache(maxsize=1)
def load_profile(
    profile: Profile, device: str = "cpu", truncate: bool = False
) -> tuple[TransformerBridge, SAE]:
    """Load a profile's model and SAE on one device, keeping only the latest pair cached.

    truncate=True drops the blocks after the SAE's layer: less memory and the same activations
    at the SAE, but no generation, so only offline extraction uses it. The cached pair is
    released only after the next one has loaded; call load_profile.cache_clear() first when two
    would not fit in memory together.
    """
    model = TransformerBridge.boot_transformers(
        profile.model_name,
        device=device,
        dtype=DTYPES[profile.dtype],
        hf_config_overrides=truncation(profile) if truncate else None,
    )
    # Compatibility mode gives the blocks.N.hook_* names the SAEs use.
    model.enable_compatibility_mode(no_processing=not profile.process_weights)
    sae = SAE.from_pretrained(profile.sae_release, profile.sae_id, device=device)
    return model, sae


def load_model_and_sae(device: str = "cpu") -> tuple[TransformerBridge, SAE]:
    """Load the active profile's model and SAE once per process."""
    return load_profile(active_profile(), device)


def main() -> None:
    """Print the loaded model and SAE dimensions."""
    profile = active_profile()
    model, sae = load_model_and_sae()
    print(f"Profile: {profile.name} ({profile.model_name}, {profile.dtype})")
    print(f"Device: {model.cfg.device}")
    print(f"Hook: {sae.cfg.metadata.hook_name}")
    print(f"d_model: {model.cfg.d_model}")
    print(f"d_sae: {sae.cfg.d_sae}")


if __name__ == "__main__":
    main()
