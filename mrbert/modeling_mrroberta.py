"""
MrRoBERTa: RoBERTa with Dynamic Token Merging (DeleteGate).

This mirrors the MrBERT / MrXLM designs but wraps a HuggingFace RobertaModel
backbone. The goal is architectural: same gate + encoder contract on RoBERTa.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from transformers import RobertaModel, RobertaConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .modeling_mrbert import DeleteGate, softmax1, _run_layer_with_soft_gate


class MrRobertaModel(RobertaModel):
    """
    RoBERTa with a delete gate after a fixed encoder layer.

    Contract matches MrBertModel / MrXLMRobertaModel:
      _encoder_forward_with_gate(...) -> (hidden_states, gate, keep_indices, kept_lengths)
    so QA coordinate remapping and diagnostics can be reused.
    """

    config_class = RobertaConfig

    def __init__(self, config: RobertaConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.mrroberta_config = config

        self.gate_k: float = getattr(config, "gate_k", -30.0)
        self.gate_layer_index: int = getattr(config, "gate_layer_index", 3)
        self.gate_threshold_ratio: float = getattr(config, "gate_threshold_ratio", 0.5)
        self.use_softmax1: bool = getattr(config, "use_softmax1", True)
        self.force_keep_cls: bool = getattr(config, "force_keep_cls", True)
        self.log_shapes: bool = getattr(config, "log_shapes", False)

        if not hasattr(config, "gate_k"):
            config.gate_k = self.gate_k

        self.delete_gate = DeleteGate(config)  # type: ignore[arg-type]

    @classmethod
    def from_pretrained_roberta(
        cls,
        pretrained_name_or_path: str,
        *,
        gate_layer_index: int = 3,
        gate_k: float = -30.0,
        gate_threshold_ratio: float = 0.5,
        use_softmax1: bool = True,
        force_keep_cls: bool = True,
        **kwargs,
    ) -> "MrRobertaModel":
        """
        Load a standard RoBERTa backbone and attach gate hyperparameters.
        """
        base = RobertaModel.from_pretrained(pretrained_name_or_path, **kwargs)
        config = base.config
        config.gate_layer_index = gate_layer_index
        config.gate_k = gate_k
        config.gate_threshold_ratio = gate_threshold_ratio
        config.use_softmax1 = use_softmax1
        config.force_keep_cls = force_keep_cls

        model = cls(config)
        model.load_state_dict(base.state_dict(), strict=False)
        return model

    def get_gate_regularizer_loss(self, gate: torch.Tensor) -> torch.Tensor:
        return gate.mean()

    def _encoder_forward_with_gate(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        head_mask: torch.Tensor | None,
        use_soft_deletion: bool,
        return_gate: bool = False,
    ):
        encoder = self.encoder
        layers = encoder.layer
        gate_layer_index = self.gate_layer_index

        keep_indices_out: torch.Tensor | None = None
        kept_lengths_out: torch.Tensor | None = None

        for i in range(gate_layer_index + 1):
            layer = layers[i]
            layer_outputs = layer(hidden_states, attention_mask=attention_mask, head_mask=head_mask[i] if head_mask else None)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        gate = self.delete_gate(hidden_states)  # (batch, seq_len)

        if use_soft_deletion:
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
            threshold = self.gate_k * self.gate_threshold_ratio
            batch, seq_len, hidden_size = hidden_states.shape
            device = hidden_states.device

            keep_masks = gate > threshold
            if self.force_keep_cls and seq_len > 0:
                keep_masks[:, 0] = True

            kept_lengths = keep_masks.sum(dim=1)
            max_kept = kept_lengths.max().item()
            if max_kept == 0:
                max_kept = 1

            keep_indices = torch.zeros(batch, max_kept, dtype=torch.long, device=device)
            for b in range(batch):
                idx = torch.where(keep_masks[b])[0]
                keep_indices[b, : len(idx)] = idx
                if len(idx) < max_kept:
                    keep_indices[b, len(idx) :] = idx[-1] if len(idx) > 0 else 0

            keep_indices_out = keep_indices
            kept_lengths_out = kept_lengths

            if self.log_shapes:
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

        # RoBERTa does not use token_type_ids but keep for interface symmetry.
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
            outputs = (hidden_states, pooled_output)
            if output_hidden_states:
                outputs = outputs + (None,)
            if output_attentions:
                outputs = outputs + (None,)
            if return_gate:
                outputs = outputs + (gate, keep_indices, kept_lengths)
            return outputs  # type: ignore[return-value]

        model_output = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=None,
            past_key_values=None,
            attentions=None,
            cross_attentions=None,
        )

        if return_gate:
            model_output.gate = gate  # type: ignore[attr-defined]
            model_output.keep_indices = keep_indices  # type: ignore[attr-defined]
            model_output.kept_lengths = kept_lengths  # type: ignore[attr-defined]

        return model_output

