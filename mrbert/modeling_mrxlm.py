"""
MrXLM: XLM-R with Dynamic Token Merging (DeleteGate + PI-ready interface).

This module mirrors the MrBERT design but wraps a HuggingFace XLM-R backbone.
The goal is architectural: show that the delete gate and encoder wiring are
decoupled from the specific Transformer implementation.

We intentionally keep this lightweight:
- Reuse the existing DeleteGate and soft-gated encoder logic from MrBERT.
- Provide a drop-in XLM-R encoder with the same (hidden_states, gate,
  keep_indices, kept_lengths) contract used by the QA index remapping code.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from transformers import XLMRobertaModel, XLMRobertaConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

# Reuse gate and soft-deletion helper from the MrBERT implementation.
from .modeling_mrbert import DeleteGate, softmax1, _run_layer_with_soft_gate


class MrXLMRobertaModel(XLMRobertaModel):
    """
    XLM-R with a delete gate after a fixed encoder layer.

    Design goals:
    - Minimal changes on top of HuggingFace XLMRobertaModel.
    - Same gate behavior and encoder contract as MrBertModel:
      _encoder_forward_with_gate(...) -> (hidden_states, gate, keep_indices, kept_lengths).
    - No task heads here; downstream code can wrap this model similarly to MrBertFor*.
    """

    config_class = XLMRobertaConfig

    def __init__(self, config: XLMRobertaConfig, *args, **kwargs) -> None:
        # Allow passing gate hyperparameters via config extras:
        #   gate_k: float in [-30, 0] (default -30.0)
        #   gate_layer_index: encoder layer index where gate is placed (default 3)
        #   gate_threshold_ratio: threshold = gate_k * ratio for hard deletion (default 0.5)
        #   use_softmax1: whether to use softmax1 vs softmax for attention (default True)
        #   force_keep_cls: always keep first token during hard deletion (default True)
        super().__init__(config, *args, **kwargs)

        self.mrxlm_config = config

        # Defaults, but allow overriding via config attributes.
        self.gate_k: float = getattr(config, "gate_k", -30.0)
        self.gate_layer_index: int = getattr(config, "gate_layer_index", 3)
        self.gate_threshold_ratio: float = getattr(config, "gate_threshold_ratio", 0.5)
        self.use_softmax1: bool = getattr(config, "use_softmax1", True)
        self.force_keep_cls: bool = getattr(config, "force_keep_cls", True)
        self.log_shapes: bool = getattr(config, "log_shapes", False)

        # Make sure DeleteGate sees the gate_k we just set.
        if not hasattr(config, "gate_k"):
            config.gate_k = self.gate_k

        # Reuse the existing DeleteGate implementation; it only depends on
        # hidden_size, layer_norm_eps, and gate_k, which XLM-R's config provides.
        self.delete_gate = DeleteGate(config)  # type: ignore[arg-type]

    @classmethod
    def from_pretrained_xlm(
        cls,
        pretrained_name_or_path: str,
        *,
        gate_layer_index: int = 3,
        gate_k: float = -30.0,
        gate_threshold_ratio: float = 0.5,
        use_softmax1: bool = True,
        force_keep_cls: bool = True,
        **kwargs,
    ) -> "MrXLMRobertaModel":
        """
        Convenience constructor:
        - Load a standard XLM-R backbone.
        - Attach gate-specific hyperparameters to the config.
        - Instantiate MrXLMRobertaModel and load backbone weights (strict=False).

        This mirrors MrBertModel.from_pretrained_bert but for XLM-R.
        """
        base = XLMRobertaModel.from_pretrained(pretrained_name_or_path, **kwargs)
        config = base.config
        config.gate_layer_index = gate_layer_index
        config.gate_k = gate_k
        config.gate_threshold_ratio = gate_threshold_ratio
        config.use_softmax1 = use_softmax1
        config.force_keep_cls = force_keep_cls

        model = cls(config)
        # Load all XLM-R weights into the MrXLM encoder; gate params are new.
        model.load_state_dict(base.state_dict(), strict=False)
        return model

    def get_gate_regularizer_loss(self, gate: torch.Tensor) -> torch.Tensor:
        """Same definition as MrBertModel: L_G = mean(G)."""
        return gate.mean()

    def _encoder_forward_with_gate(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        head_mask: torch.Tensor | None,
        use_soft_deletion: bool,
        return_gate: bool = False,
    ):
        """
        Returns (hidden_states, gate, keep_indices, kept_lengths).

        When use_soft_deletion=True or return_gate=False, keep_indices and kept_lengths are None.
        When use_soft_deletion=False (hard deletion), keep_indices[b, j] is the original token
        index of the j-th kept token in batch b; kept_lengths[b] is the number of kept tokens.
        This matches the MrBertModel contract used by QA coordinate remapping code.
        """
        encoder = self.encoder
        layers = encoder.layer
        gate_layer_index = self.gate_layer_index

        keep_indices_out: torch.Tensor | None = None
        kept_lengths_out: torch.Tensor | None = None

        # Layers 0 .. gate_layer (inclusive)
        for i in range(gate_layer_index + 1):
            layer = layers[i]
            layer_outputs = layer(hidden_states, attention_mask=attention_mask, head_mask=head_mask[i] if head_mask else None)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        # Delete gate after layer gate_layer_index
        gate = self.delete_gate(hidden_states)  # (batch, seq_len)

        if use_soft_deletion:
            # Soft deletion: run remaining layers with gate added to attention.
            for i in range(gate_layer_index + 1, len(layers)):
                hidden_states = _run_layer_with_soft_gate(
                    layers[i],
                    hidden_states,
                    attention_mask,
                    gate,
                    self.gate_k,
                    self.use_softmax1,
                    None,
                )
        else:
            # Hard deletion: keep tokens where G > threshold, shorten sequence, update mask.
            threshold = self.gate_k * self.gate_threshold_ratio
            batch, seq_len, hidden_size = hidden_states.shape
            device = hidden_states.device

            keep_masks = gate > threshold  # (batch, seq_len)

            # XLM-R also uses a leading special token (equivalent to [CLS]); keep it if requested.
            if self.force_keep_cls and seq_len > 0:
                keep_masks[:, 0] = True

            kept_lengths = keep_masks.sum(dim=1)  # (batch,)
            max_kept = kept_lengths.max().item()
            if max_kept == 0:
                max_kept = 1

            # Build indices: for each batch item, indices of kept positions (original sequence positions).
            keep_indices = torch.zeros(batch, max_kept, dtype=torch.long, device=device)
            for b in range(batch):
                idx = torch.where(keep_masks[b])[0]
                keep_indices[b, : len(idx)] = idx
                if len(idx) < max_kept:
                    keep_indices[b, len(idx) :] = idx[-1] if len(idx) > 0 else 0

            keep_indices_out = keep_indices
            kept_lengths_out = kept_lengths

            # Gather hidden states: (batch, max_kept, hidden_size)
            if self.log_shapes:
                # Reuse the diagnostics helper if available.
                try:
                    from .diagnostics import log_shape_after_gate

                    log_shape_after_gate(batch, seq_len, max_kept, hidden_size, gate_layer_index, num_layers=len(layers))
                except Exception:
                    pass

            hidden_states = torch.gather(
                hidden_states,
                1,
                keep_indices.unsqueeze(-1).expand(-1, -1, hidden_size),
            )

            # New attention mask: 1 for real tokens, 0 for padding.
            new_attention_mask = (torch.arange(max_kept, device=device).unsqueeze(0) < kept_lengths.unsqueeze(1)).to(
                hidden_states.dtype
            )
            new_attention_mask = (1.0 - new_attention_mask) * -10000.0
            new_attention_mask = new_attention_mask.unsqueeze(1).unsqueeze(2)

            for i in range(gate_layer_index + 1, len(layers)):
                layer_outputs = layers[i](hidden_states, attention_mask=new_attention_mask)
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        if return_gate:
            return hidden_states, gate, keep_indices_out, kept_lengths_out
        return hidden_states, None, None, None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        use_soft_deletion: bool | None = None,
        return_gate: bool = False,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        """
        Forward pass with optional soft/hard deletion.

        This closely follows XLMRobertaModel.forward but inserts the gate after
        a fixed encoder layer and (optionally) returns gate / index metadata.
        """
        if use_soft_deletion is None:
            use_soft_deletion = self.training

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        # XLM-R does not use token_type_ids, but keep the argument for interface symmetry.
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers) if head_mask is not None else None

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
        )

        hidden_states, gate, keep_indices, kept_lengths = self._encoder_forward_with_gate(
            embedding_output,
            extended_attention_mask,
            head_mask,
            use_soft_deletion=use_soft_deletion,
            return_gate=return_gate,
        )

        pooled_output = self.pooler(hidden_states) if self.pooler is not None else None

        if not return_dict:
            # Match HuggingFace's tuple layout: (last_hidden_state, pooler_output, hidden_states, attentions, ...)
            outputs = (hidden_states, pooled_output)
            if output_hidden_states:
                outputs = outputs + (None,)
            if output_attentions:
                outputs = outputs + (None,)
            if return_gate:
                outputs = outputs + (gate, keep_indices, kept_lengths)
            return outputs  # type: ignore[return-value]

        # For now we do not populate hidden_states/attentions; callers that need them
        # should rely on MrBertModel for detailed analysis.
        model_output = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=None,
            past_key_values=None,
            attentions=None,
            cross_attentions=None,
        )

        if return_gate:
            # Attach gate metadata as attributes for downstream consumers.
            model_output.gate = gate  # type: ignore[attr-defined]
            model_output.keep_indices = keep_indices  # type: ignore[attr-defined]
            model_output.kept_lengths = kept_lengths  # type: ignore[attr-defined]

        return model_output

