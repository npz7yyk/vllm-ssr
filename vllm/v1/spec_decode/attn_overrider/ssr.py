import torch

from vllm.config import VllmConfig
from .abstract import AbstractAttentionOverrider, override_return


class SSRAttentionOverrider(AbstractAttentionOverrider):
    """An attention overrider that implements SSR sparse attention."""

    # Wether this overrider uses a private attention implementation.
    use_private_attention = True

    def __init__(
        self,
        # Base class args.
        vllm_config: VllmConfig,
        device: torch.device,
        # Sliding window specific args.
        **_kwargs
    ):
        super().__init__(vllm_config, device)

    @override_return(use_private_attention)
    def __call__(self, layer_index: int, *_args, **_kwargs):
        return
