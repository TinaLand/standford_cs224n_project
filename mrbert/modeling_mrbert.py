"""
MrBERT: BERT with Dynamic Token Merging (adapted from MrT5, ICLR 2025).

After a fixed encoder layer, a learned delete gate selects which tokens to keep.
Training: soft deletion (gate as attention mask). Inference: hard deletion (column removal).
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, BertConfig, BertLayer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .configuration_mrbert import MrBertConfig


def softmax1(x, dim=-1):
    """Softmax1 from paper Eq.(7): (softmax1(x))_i = exp(x_i) / (1 + sum_j exp(x_j)).
    Used so that when all G_i = k, attention weights do not collapse."""
    exp_x = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


class DeleteGate(nn.Module):
    """Delete gate, paper Eq.(1): G = k * sigma(LayerNorm(H_l) W + 1_N b), output in [k, 0].
    Only 2*d_model + 1 extra parameters (LayerNorm, W, b)."""

    def __init__(self, config: MrBertConfig):
        super().__init__()
        self.gate_k = config.gate_k
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_size)
        x = self.layer_norm(hidden_states)
        logits = self.linear(x).squeeze(-1) + self.bias  # (batch, seq_len)
        g = self.gate_k * torch.sigmoid(logits)  # (batch, seq_len), in [k, 0]
        return g


def _run_layer_with_soft_gate(
    layer: BertLayer,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    gate: torch.Tensor,
    gate_k: float,
    use_softmax1: bool,
    head_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one BERT layer with soft deletion (paper Section 3.1, Eq.(2)):
    attention_scores = QK^T/sqrt(d) + attention_mask + 1_N G^T, then softmax1 (or softmax) and multiply by V.
    Fully differentiable; sequence length unchanged during training."""
    # Self-attention with gate
    self_attn = layer.attention.self
    batch, seq_len, _ = hidden_states.shape
    mixed_query = self_attn.query(hidden_states)
    mixed_key = self_attn.key(hidden_states)
    mixed_value = self_attn.value(hidden_states)
    num_heads = self_attn.num_attention_heads
    head_dim = self_attn.attention_head_size
    query = mixed_query.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    key = mixed_key.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    value = mixed_value.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    # query, key, value: (batch, num_heads, seq_len, head_dim)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim**0.5)
    # attention_scores: (batch, num_heads, seq_len, seq_len)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask
    # Add gate: mask key positions (dim -1). G has shape (batch, seq_len); add (1,1,1,seq_len)
    gate_broadcast = gate.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
    attention_scores = attention_scores + gate_broadcast
    if use_softmax1:
        attention_probs = softmax1(attention_scores, dim=-1)
    else:
        attention_probs = F.softmax(attention_scores, dim=-1)
    if head_mask is not None:
        attention_probs = attention_probs * head_mask
    context = torch.matmul(attention_probs, value)
    context = context.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)
    attn_output = layer.attention.output.dense(context)
    attn_output = layer.attention.output.LayerNorm(attn_output + hidden_states)
    # FFN
    ffn_output = layer.intermediate(attn_output)
    ffn_output = layer.output(ffn_output, attn_output)
    return ffn_output


class MrBertModel(BertModel):
    """BERT with dynamic token merging (delete gate after a fixed encoder layer)."""

    config_class = MrBertConfig

    def __init__(self, config: MrBertConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.mrbert_config = config
        self.delete_gate = DeleteGate(config)
        self.gate_layer_index = config.gate_layer_index
        self.gate_k = config.gate_k
        self.gate_threshold_ratio = config.gate_threshold_ratio
        self.use_softmax1 = getattr(config, "use_softmax1", True)

    @classmethod
    def from_pretrained_bert(cls, bert_name_or_path: str, **mrbert_config_kwargs):
        """Load BERT weights into MrBERT; gate params are randomly initialized."""
        bert = BertModel.from_pretrained(bert_name_or_path)
        config = MrBertConfig(**{**bert.config.to_dict(), **mrbert_config_kwargs})
        model = cls(config)
        model.load_state_dict(bert.state_dict(), strict=False)
        return model

    def get_gate_regularizer_loss(self, gate: torch.Tensor) -> torch.Tensor:
        """Gate regularizer, paper Eq.(3): L_G = (1/N) * sum_i G_i; total loss L = L_CE + alpha * L_G."""
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
        gate_layer_index = self.gate_layer_index
        layers = encoder.layer

        # Layers 0 .. gate_layer (inclusive)
        for i in range(gate_layer_index + 1):
            layer = layers[i]
            layer_outputs = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        # Delete gate after layer gate_layer_index
        gate = self.delete_gate(hidden_states)  # (batch, seq_len)

        if use_soft_deletion:
            # Soft deletion: run remaining layers with gate added to attention
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
            # Hard deletion (paper Section 3.1): keep tokens where G > threshold, shorten sequence, update mask
            threshold = self.gate_k * self.gate_threshold_ratio
            batch, seq_len, hidden_size = hidden_states.shape
            device = hidden_states.device
            keep_masks = gate > threshold  # (batch, seq_len)
            # BERT: force-keep [CLS] (index 0) so pooler and classification head get a valid token
            if getattr(self.mrbert_config, "force_keep_cls", True):
                keep_masks[:, 0] = True
            kept_lengths = keep_masks.sum(dim=1)  # (batch,)
            max_kept = kept_lengths.max().item()
            if max_kept == 0:
                max_kept = 1
            # Build indices: for each batch item, indices of kept positions
            keep_indices = torch.zeros(batch, max_kept, dtype=torch.long, device=device)
            for b in range(batch):
                idx = torch.where(keep_masks[b])[0]
                keep_indices[b, : len(idx)] = idx
                if len(idx) < max_kept:
                    keep_indices[b, len(idx) :] = idx[-1] if len(idx) > 0 else 0
            # Gather hidden states: (batch, max_kept, hidden_size)
            hidden_states = torch.gather(
                hidden_states,
                1,
                keep_indices.unsqueeze(-1).expand(-1, -1, hidden_size),
            )
            # New attention mask: 1 for real tokens, 0 for padding
            new_attention_mask = (torch.arange(max_kept, device=device).unsqueeze(0) < kept_lengths.unsqueeze(1)).to(
                hidden_states.dtype
            )
            new_attention_mask = (1.0 - new_attention_mask) * -10000.0
            new_attention_mask = new_attention_mask.unsqueeze(1).unsqueeze(2)
            for i in range(gate_layer_index + 1, len(layers)):
                layer_outputs = layers[i](hidden_states, attention_mask=new_attention_mask)
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        if return_gate:
            return hidden_states, gate
        return hidden_states, None

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
        past_key_values: list | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        use_soft_deletion: bool | None = None,
        return_gate: bool = False,
    ):
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
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        if head_mask is not None and hasattr(self, "get_head_mask"):
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        elif head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
        )

        hidden_states, gate = self._encoder_forward_with_gate(
            embedding_output,
            extended_attention_mask,
            head_mask,
            use_soft_deletion=use_soft_deletion,
            return_gate=return_gate,
        )

        pooled_output = self.pooler(hidden_states) if self.pooler is not None else None

        if not return_dict:
            out = (hidden_states, pooled_output)
            if return_gate and gate is not None:
                return out + (gate,)
            return out

        outputs = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )
        if return_gate and gate is not None:
            return outputs, gate
        return outputs

    def forward_with_gate_loss(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        gate_regularizer_weight: float = 1e-4,
        **kwargs,
    ):
        """Forward and return (outputs, gate_regularizer_loss)."""
        result = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_gate=True,
            use_soft_deletion=True,
            **kwargs,
        )
        if isinstance(result, tuple):
            outputs, gate = result
        else:
            outputs, gate = result, None
        gate_loss = self.get_gate_regularizer_loss(gate) if gate is not None else torch.tensor(0.0, device=next(self.parameters()).device)
        return outputs, gate_loss * gate_regularizer_weight


class MrBertForSequenceClassification(nn.Module):
    """MrBERT with a classification head on top (e.g. for GLUE)."""

    def __init__(self, config: MrBertConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.mrbert = MrBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    @classmethod
    def from_bert_pretrained(cls, bert_name_or_path: str, num_labels: int = 2, **mrbert_kwargs):
        """Load BERT weights and add MrBERT gate (gate params randomly initialized)."""
        from transformers import BertForSequenceClassification

        bert_model = BertForSequenceClassification.from_pretrained(bert_name_or_path, num_labels=num_labels)
        bert_config = bert_model.config
        config = MrBertConfig(**{**bert_config.to_dict(), **mrbert_kwargs})
        model = cls(config, num_labels=num_labels)
        model.mrbert.load_state_dict(bert_model.bert.state_dict(), strict=False)
        model.classifier.load_state_dict(bert_model.classifier.state_dict())
        model.dropout.load_state_dict(bert_model.dropout.state_dict())
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
        result = self.mrbert.forward(
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
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        # Paper Eq.(3): L = L_CE + alpha * L_G; alpha = gate_regularizer_weight (or from PI controller)
        gate_loss = self.mrbert.get_gate_regularizer_loss(gate) * gate_regularizer_weight if gate is not None else 0.0
        if loss is not None and gate_loss != 0:
            loss = loss + gate_loss
        return dict(loss=loss, logits=logits, gate=gate)


class MrBertForQuestionAnswering(nn.Module):
    """MrBERT with a span (QA) head on top for extractive QA (e.g. SQuAD, TyDi QA)."""

    def __init__(self, config: MrBertConfig):
        super().__init__()
        self.config = config
        self.mrbert = MrBertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # start and end logits per token

    @classmethod
    def from_bert_pretrained(cls, bert_name_or_path: str, **mrbert_kwargs):
        """Load BERT weights and add MrBERT gate; QA head from BertForQuestionAnswering."""
        from transformers import BertForQuestionAnswering

        bert_qa = BertForQuestionAnswering.from_pretrained(bert_name_or_path)
        bert_config = bert_qa.config
        config = MrBertConfig(**{**bert_config.to_dict(), **mrbert_kwargs})
        model = cls(config)
        model.mrbert.load_state_dict(bert_qa.bert.state_dict(), strict=False)
        model.qa_outputs.load_state_dict(bert_qa.qa_outputs.state_dict())
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
        result = self.mrbert.forward(
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
        sequence_output = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        logits = self.qa_outputs(sequence_output)  # (batch, seq_len, 2)
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]
        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
        gate_loss = self.mrbert.get_gate_regularizer_loss(gate) * gate_regularizer_weight if gate is not None else 0.0
        if loss is not None and gate_loss != 0:
            loss = loss + gate_loss
        return dict(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            gate=gate,
        )
