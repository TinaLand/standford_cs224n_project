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


class XLMRobertaClassificationHead(nn.Module):
    """Classification head matching HuggingFace XLMRobertaForSequenceClassification."""

    def __init__(self, config: XLMRobertaConfig, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Features are already pooled (batch_size, hidden_size).
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MrXLMRobertaForSequenceClassification(nn.Module):
    """XLM-R with delete gate and a classification head (for NLI, sentiment, etc.)."""

    def __init__(self, config: XLMRobertaConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.mrxlm = MrXLMRobertaModel(config)
        # Keep a top-level dropout to match HuggingFace module structure.
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Use a head whose submodules (dense / dropout / out_proj) match HF state_dict keys.
        self.classifier = XLMRobertaClassificationHead(config, num_labels)

    @classmethod
    def from_pretrained_xlm(
        cls,
        pretrained_name_or_path: str,
        num_labels: int = 2,
        *,
        gate_layer_index: int = 3,
        gate_k: float = -30.0,
        **kwargs,
    ) -> "MrXLMRobertaForSequenceClassification":
        """Load XLM-R weights and add MrXLM gate; classifier from XLMRobertaForSequenceClassification."""
        from transformers import XLMRobertaForSequenceClassification as HFXLMRobertaForSequenceClassification

        hf_model = HFXLMRobertaForSequenceClassification.from_pretrained(
            pretrained_name_or_path, num_labels=num_labels, **kwargs
        )
        config = hf_model.config
        config.gate_layer_index = gate_layer_index
        config.gate_k = gate_k
        config.gate_threshold_ratio = getattr(config, "gate_threshold_ratio", 0.5)

        model = cls(config, num_labels=num_labels)
        model.mrxlm.load_state_dict(hf_model.roberta.state_dict(), strict=False)
        model.classifier.load_state_dict(hf_model.classifier.state_dict())
        return model

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        gate_regularizer_weight: float = 0.0,
        **kwargs,
    ):
        result = self.mrxlm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_gate=True,
            use_soft_deletion=self.training,
            **kwargs,
        )
        if isinstance(result, tuple):
            outputs, gate = result
        else:
            outputs, gate = result, None
        pooler_output = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[1]
        logits = self.classifier(self.dropout(pooler_output))
        loss_ce = None
        if labels is not None:
            loss_ce = nn.functional.cross_entropy(logits, labels)
        loss_gate = self.mrxlm.get_gate_regularizer_loss(gate) * gate_regularizer_weight if gate is not None else None
        loss = (loss_ce + loss_gate) if (loss_ce is not None and loss_gate is not None) else (loss_ce if loss_ce is not None else None)
        return dict(loss=loss, logits=logits, gate=gate, loss_ce=loss_ce, loss_gate=loss_gate)


class MrXLMRobertaForQuestionAnswering(nn.Module):
    """XLM-R with delete gate and a span (QA) head for extractive QA (e.g. TyDi QA)."""

    def __init__(self, config: XLMRobertaConfig):
        super().__init__()
        self.config = config
        self.mrxlm = MrXLMRobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # start and end logits per token

    @classmethod
    def from_pretrained_xlm(
        cls,
        pretrained_name_or_path: str,
        *,
        gate_layer_index: int = 3,
        gate_k: float = -30.0,
        gate_threshold_ratio: float = 0.5,
        **kwargs,
    ) -> "MrXLMRobertaForQuestionAnswering":
        """Load XLM-R weights and add MrXLM gate; QA head from XLMRobertaForQuestionAnswering."""
        from transformers import XLMRobertaForQuestionAnswering as HFXLMRobertaForQuestionAnswering

        hf_qa = HFXLMRobertaForQuestionAnswering.from_pretrained(pretrained_name_or_path, **kwargs)
        config = hf_qa.config
        config.gate_layer_index = gate_layer_index
        config.gate_k = gate_k
        config.gate_threshold_ratio = gate_threshold_ratio

        model = cls(config)
        model.mrxlm.load_state_dict(hf_qa.roberta.state_dict(), strict=False)
        model.qa_outputs.load_state_dict(hf_qa.qa_outputs.state_dict())
        return model

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
        gate_regularizer_weight: float = 0.0,
        **kwargs,
    ):
        result = self.mrxlm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_gate=True,
            use_soft_deletion=self.training,
            **kwargs,
        )
        if isinstance(result, tuple):
            outputs, gate = result
        else:
            outputs, gate = result, None
        keep_indices = getattr(outputs, "keep_indices", None)
        kept_lengths = getattr(outputs, "kept_lengths", None)
        sequence_output = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        logits = self.qa_outputs(sequence_output)  # (batch, seq_len, 2)
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]
        loss_ce = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss_ce = (start_loss + end_loss) / 2
        loss_gate = self.mrxlm.get_gate_regularizer_loss(gate) * gate_regularizer_weight if gate is not None else None
        loss = (loss_ce + loss_gate) if (loss_ce is not None and loss_gate is not None) else (loss_ce if loss_ce is not None else None)
        out = dict(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            gate=gate,
            loss_ce=loss_ce,
            loss_gate=loss_gate,
        )
        if keep_indices is not None:
            out["keep_indices"] = keep_indices
            out["kept_lengths"] = kept_lengths
        return out
