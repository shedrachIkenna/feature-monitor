"""Load GPT-2 Small and its block-8 residual-stream SAE."""

from functools import lru_cache

from sae_lens import SAE
from transformer_lens.model_bridge import TransformerBridge

MODEL_NAME = "gpt2"
SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.8.hook_resid_pre"


@lru_cache(maxsize=1)
def load_model_and_sae() -> tuple[TransformerBridge, SAE]:
    """Load GPT-2 Small and its block-8 residual-pre SAE once per process."""
    device = "cpu"
    model = TransformerBridge.boot_transformers(MODEL_NAME, device=device)
    model.enable_compatibility_mode()
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=device)
    return model, sae


def main() -> None:
    """Print the loaded model and SAE dimensions."""
    model, sae = load_model_and_sae()
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {model.cfg.device}")
    print(f"d_model: {model.cfg.d_model}")
    print(f"d_sae: {sae.cfg.d_sae}")


if __name__ == "__main__":
    main()
