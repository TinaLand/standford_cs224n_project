"""
MrBERT (Merge BERT) configuration.
Adapts MrT5's dynamic token merging to BERT (encoder-only).
Paper: MrT5 (Kallini et al., ICLR 2025), Section 3.
"""
from transformers import BertConfig


class MrBertConfig(BertConfig):
    """Config for MrBERT: BERT with dynamic token deletion (merge) gate.

    Paper mapping:
    - gate_layer_index: layer l after which the delete gate is applied (Section 3, Fig.4).
    - gate_k: constant k in Eq.(1), G in [k, 0]; paper uses k = -30.
    - gate_threshold_ratio: hard-deletion threshold = gate_k * this (e.g. 0.5 => k/2).
    - use_softmax1: use softmax1 in Eq.(7) for attention after gate (Section 3.2).
    - target_deletion_ratio: target fraction of tokens to delete (for PI controller, Section 3.2).
    """

    model_type = "mrbert"

    def __init__(
        self,
        gate_layer_index: int = 3,
        gate_k: float = -30.0,
        gate_threshold_ratio: float = 0.5,
        use_softmax1: bool = True,
        target_deletion_ratio: float = 0.5,
        pi_kp: float = 0.5,
        pi_ki: float = 1e-5,
        pi_gamma: float = 0.9,
        force_keep_cls: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gate_layer_index = gate_layer_index
        self.gate_k = gate_k
        # Hard deletion threshold = gate_k * gate_threshold_ratio (e.g. k/2)
        self.gate_threshold_ratio = gate_threshold_ratio
        self.use_softmax1 = use_softmax1
        # PI controller (Section 3.2, Eq.4-6): target deletion ratio and P-I gains
        self.target_deletion_ratio = target_deletion_ratio
        self.pi_kp = pi_kp
        self.pi_ki = pi_ki
        self.pi_gamma = pi_gamma
        # BERT-specific: always keep [CLS] token (index 0) during hard deletion
        self.force_keep_cls = force_keep_cls
