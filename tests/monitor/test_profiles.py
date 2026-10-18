import pytest
import torch
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

from monitor import labels, model
from monitor.baseline import baseline_path
from monitor.calibrate import threshold_path
from monitor.embed_drift import embed_baseline_path, embeddings_cache_path
from monitor.embed_mean import mean_baseline_path
from monitor.model import PROFILE_ENV, active_profile, load_profiles


@pytest.fixture
def fresh_load_cache():
    """Empty the loader cache around a test, so fakes never leak into other tests."""
    model._load.cache_clear()
    yield
    model._load.cache_clear()


def test_default_profile_is_gpt2_small(monkeypatch) -> None:
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    assert active_profile().name == "gpt2_small"


def test_environment_variable_selects_the_profile(monkeypatch) -> None:
    monkeypatch.setenv(PROFILE_ENV, "gemma2_2b")
    profile = active_profile()
    assert profile.model_name == "google/gemma-2-2b"
    assert profile.dtype == "bfloat16"
    assert profile.process_weights is False


def test_unknown_profile_is_refused(monkeypatch) -> None:
    monkeypatch.setenv(PROFILE_ENV, "gpt5")
    with pytest.raises(ValueError, match="gpt2_small"):
        active_profile()


def test_bad_dtype_and_unknown_keys_are_refused(tmp_path) -> None:
    path = tmp_path / "profiles.yaml"
    fields = {
        "label": "x",
        "model_name": "m",
        "dtype": "float16",
        "process_weights": True,
        "sae_release": "r",
        "sae_id": "i",
        "layer": 1,
        "neuronpedia_id": "m/1",
    }
    path.write_text("p:\n" + "".join(f"  {k}: {v}\n" for k, v in fields.items()))
    with pytest.raises(ValueError, match="dtype"):
        load_profiles(path)
    path.write_text(path.read_text() + "  extra: 1\n")
    with pytest.raises(TypeError):
        load_profiles(path)


@pytest.mark.parametrize("profile", load_profiles().values(), ids=lambda p: p.name)
def test_profile_matches_the_installed_sae_lens_registry(profile) -> None:
    release = get_pretrained_saes_directory()[profile.sae_release]
    assert profile.sae_id in release.saes_map
    assert release.neuronpedia_id[profile.sae_id] == profile.neuronpedia_id


def test_file_stems_keep_profiles_apart() -> None:
    profiles = load_profiles()
    assert profiles["gpt2_small"].file_stem == "gpt2_8"  # the committed baseline's name
    assert profiles["gemma2_2b"].file_stem == "gemma-2-2b_16"


def test_baselines_thresholds_and_labels_are_per_profile(monkeypatch) -> None:
    paths = {}
    for name in ("gpt2_small", "gemma2_2b"):
        monkeypatch.setenv(PROFILE_ENV, name)
        paths[name] = (baseline_path(), threshold_path(baseline_path()), labels._feature_url(7))
    assert paths["gpt2_small"][0].name == "gpt2_8.json"
    assert paths["gemma2_2b"][1].name == "gemma-2-2b_16_threshold.json"
    assert paths["gemma2_2b"][2].endswith("/gemma-2-2b/16-gemmascope-res-16k/7")
    for a, b in zip(paths["gpt2_small"], paths["gemma2_2b"], strict=True):
        assert a != b


def test_label_cache_is_named_after_the_profile_in_use() -> None:
    assert labels.CACHE_PATH.name == f"labels_cache_{active_profile().name}.json"


def test_embedding_files_are_shared_by_all_profiles(monkeypatch) -> None:
    """The embedding detector reads prompt text only, so its files must not follow the profile."""
    paths = []
    for name in ("gpt2_small", "gemma2_2b"):
        monkeypatch.setenv(PROFILE_ENV, name)
        paths.append((embed_baseline_path(), embeddings_cache_path("x"), mean_baseline_path()))
    assert paths[0] == paths[1]


def test_loader_passes_each_profiles_settings(monkeypatch, fresh_load_cache) -> None:
    calls = []

    class FakeBridge:
        @staticmethod
        def boot_transformers(name, device, dtype):
            calls.append(("model", name, device, dtype))
            return FakeBridge()

        def enable_compatibility_mode(self, no_processing):
            calls.append(("no_processing", no_processing))

    class FakeSAE:
        @staticmethod
        def from_pretrained(release, sae_id, device):
            calls.append(("sae", release, sae_id, device))
            return FakeSAE()

    monkeypatch.setattr(model, "TransformerBridge", FakeBridge)
    monkeypatch.setattr(model, "SAE", FakeSAE)

    monkeypatch.setenv(PROFILE_ENV, "gemma2_2b")
    model.load_model_and_sae("mps")
    model.load_model_and_sae("mps")  # cached: no second load
    monkeypatch.setenv(PROFILE_ENV, "gpt2_small")
    model.load_model_and_sae()

    assert calls == [
        ("model", "google/gemma-2-2b", "mps", torch.bfloat16),
        ("no_processing", True),
        ("sae", "gemma-scope-2b-pt-res-canonical", "layer_16/width_16k/canonical", "mps"),
        ("model", "gpt2", "cpu", torch.float32),
        ("no_processing", False),
        ("sae", "gpt2-small-res-jb", "blocks.8.hook_resid_pre", "cpu"),
    ]
