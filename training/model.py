import torch
import torch.nn as nn
from transformers import (
        PreTrainedModel,
        BertConfig,
        BertModel,
        )
from typing import Iterable, List
from dataclasses import dataclass

@dataclass
class MultiTaskClassfierOutput:
    loss: Optional[torch.Tensor] = None
    token_losses: Optional[List[torch.Tensor]] = None
    sequence_losses: Optional[List[torch.Tensor]] = None
    token_logits: List[torch.Tensor]
    sequence_logits: List[torch.Tensor]
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None

class BertForMultiTaskClassification(PreTrainedModel):
    def __init__(self, config: BertConfig, num_token_labels: Iterable[int], num_sequence_labels: Iterable[int], token_coefs: Iterable[float], sequence_coefs: Iterable[float]):
        super().__init__(config)
        self.bert = BertModel(config)
        self.num_token_labels = num_token_labels
        self.num_sequence_labels = num_sequence_labels
        self.token_classifiers = nn.ModuleList([nn.Linear(config.hidden_size, num_labels) for num_labels in num_token_labels])
        self.sequence_classifiers = nn.ModuleList([nn.Linear(config.hidden_size, num_labels) for num_labels in num_sequence_labels])
        dropout_prob = config.classifier_dropout if config.classifier_dropout is not None else 0.
        self.dropout = nn.Dropout(dropout_prob)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(input_ids: Optional = None, attention_mask: Optional = None, token_type_ids: Optional = None, position_ids: Optional = None, head_mask: Optional = None, inputs_embeds: Optional = None, token_labels: Optional = None, sequence_labels: Optional = None, output_attentions: Optional = None, output_hidden_states: Optional = None,) -> SequenceClassifierOutput:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        # Token classification
        token_logits = [classifier(self.dropout(outputs.last_hidden_state)) for classifier in self.token_classifiers]
        loss = None
        if token_labels is not None:
            token_losses = [self.loss_fn(logit, label) for logit, label in zip(token_logits, token_labels)]
            token_loss = sum([coef * loss for coef, loss in zip(self.token_coefs, token_losses)])
            loss = token_loss

        # Sequence classification
        sequence_logits = [classifier(self.dropout(outputs.pooler_output)) for classifier in self.sequence_classifiers]
        if sequence_labels is not None:
            sequence_losses = [self.loss_fn(logit, label) for logit, label in zip(sequence_logits, sequence_labels)]
            sequence_loss = sum([coef * loss for coef, loss in zip(self.sequence_coefs, sequence_losses)])
            loss = loss + sequence_loss if loss is not None else sequence_loss

        return MultiTaskClassfierOutput(
            loss=loss,
            token_losses=token_losses,
            sequence_losses=sequence_losses,
            token_logits=token_logits,
            sequence_logits=sequence_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


