from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, PreTrainedModel


class BertForMultiTaskClassification(PreTrainedModel):
    def __init__(
        self,
        model_id: str,
        num_token_labels: int,
        num_sequence_labels: int,
        token_coef: float,
        sequence_coef: float,
        classifier_dropout: Optional[float] = None,
    ):
        config = BertConfig()
        super().__init__(config)
        self.bert = BertModel.from_pretrained(model_id)
        self.num_token_labels = num_token_labels
        self.num_sequence_labels = num_sequence_labels
        self.token_classifier = nn.Linear(config.hidden_size, self.num_token_labels)
        self.sequence_classifier = nn.Linear(config.hidden_size, self.num_sequence_labels)
        dropout_prob = classifier_dropout if classifier_dropout is not None else 0.0
        self.dropout = nn.Dropout(dropout_prob)
        self.loss_fn = nn.CrossEntropyLoss()
        self.token_coef = token_coef
        self.sequence_coef = sequence_coef

    def forward(
        self,
        input_ids: Optional = None,
        attention_mask: Optional = None,
        token_type_ids: Optional = None,
        position_ids: Optional = None,
        head_mask: Optional = None,
        inputs_embeds: Optional = None,
        token_labels: Optional = None,
        sequence_labels: Optional = None,
    ) -> dict:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        # Token classification
        token_logits = self.token_classifier(self.dropout(outputs.last_hidden_state))
        loss = None
        if token_labels is not None:
            transposed_logits = token_logits.transpose(-2, -1)
            token_loss = self.loss_fn(transposed_logits, token_labels)
            loss = token_loss * self.token_coef

        # Sequence classification
        sequence_logits = self.sequence_classifier(self.dropout(outputs.pooler_output))
        if sequence_labels is not None:
            sequence_loss = self.loss_fn(sequence_logits, sequence_labels)
            loss = loss + sequence_loss * self.sequence_coef if loss is not None else sequence_loss

        assert loss.ndim == 0, f"Loss must be a scalar, got {loss.ndim} dimensions"

        # Full logits
        logits = {"token_logits": token_logits, "sequence_logits": sequence_logits}

        output = {}
        output["logits"] = logits
        output["loss"] = loss
        output["token_loss"] = token_loss
        output["sequence_loss"] = sequence_loss

        return output
