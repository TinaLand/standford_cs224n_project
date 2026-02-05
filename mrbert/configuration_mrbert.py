"""
MrBERT (Merge BERT) configuration.
Adapts MrT5's dynamic token merging to BERT (encoder-only).
"""
from transformers import BertConfig


class MrBertConfig(BertConfig):
    """Config for MrBERT: BERT with dynamic token deletion (merge) gate."""

    model_type = "mrbert"

    def __init__(
        self,
        gate_layer_index: int = 3,
        gate_k: float = -30.0,
        gate_threshold_ratio: float = 0.5,
        use_softmax1: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gate_layer_index = gate_layer_index
        self.gate_k = gate_k
        # Hard deletion threshold = gate_k * gate_threshold_ratio (e.g. k/2)
        self.gate_threshold_ratio = gate_threshold_ratio
        self.use_softmax1 = use_softmax1
