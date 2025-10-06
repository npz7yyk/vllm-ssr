import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

type SSRConfig = dict
METHOD_KEY = "kv_sparsity"
RESERVED_KEYS = ["vllm_config", "device"]

logger = init_logger(__name__)


def build_attention_overrider(
    vllm_config: VllmConfig,
    device: torch.device,
):
    # Get the attention overrider config.
    if vllm_config.speculative_config.ssr_config is None:
        attention_overrider_config = dict()
    else:
        attention_overrider_config = vllm_config.speculative_config.ssr_config
        assert isinstance(attention_overrider_config, dict)
        # Remove reserved keys to avoid passing them twice.
        for key in RESERVED_KEYS:
            if key in attention_overrider_config:
                raise KeyError(f"Reserved key '{key}' found in ssr_config.")
    method = attention_overrider_config.get(METHOD_KEY, None)

    # Resolve the attention overrider class.
    if method is None:
        method = "ssr"
        logger.info(f"'{METHOD_KEY}' field not specified. Defaulting to 'ssr'.")
    if method == "ssr":
        from .ssr import SSRAttentionOverrider
        cls = SSRAttentionOverrider
    elif method == "sliding_window":
        from .sliding_window import SlidingWindowAttentionOverrider
        cls = SlidingWindowAttentionOverrider
    elif method == "retroinfer":
        from .retroinfer import RetroInferAttentionOverrider
        cls = RetroInferAttentionOverrider
    else:
        raise ValueError(f"Unknown SSR attention overrider: {method}")
    cls_name = cls.__name__.strip('\'')
    logger.info(f"Resolved SSR attention overrider: {cls_name}")

    # Instantiate the attention overrider.
    return cls(
        vllm_config=vllm_config,
        device=device,
        **attention_overrider_config
    )
