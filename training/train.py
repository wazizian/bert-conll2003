import os

os.environ["WANDB_PROJECT"] = "bert-mt-classif"

import math

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict
from model import BertForMultiTaskClassification
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.data.data_collator import DataCollatorMixin


def prepare_data(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    processed_dataset = DatasetDict.load_from_disk(cfg.dataset_path)
    train_dataset = processed_dataset["train"].shuffle(seed=cfg.seed).select(range(cfg.n_train))
    val_dataset = processed_dataset["validation"]
    test_dataset = processed_dataset["test"]

    return train_dataset, val_dataset, test_dataset


class DataCollator(DataCollatorMixin):
    def __init__(self, tokenizer):
        self._collator = DataCollatorForTokenClassification(tokenizer)
        self.return_tensors = "pt"

    def torch_call(self, features):
        _features = [
            {k: v for (k, v) in feature.items() if k != "sequence_labels"} for feature in features
        ]
        for feature in _features:
            feature["label"] = feature.pop("token_labels")
        _collated = self._collator(_features)
        _collated["sequence_labels"] = torch.tensor(
            [feature["sequence_labels"] for feature in features]
        )
        _collated["token_labels"] = _collated.pop("label")
        return _collated


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    tokens_logits = logits[0]["token_logits"]
    sequence_logits = logits[0]["sequence_logits"]

    token_labels = labels[0]
    sequence_labels = labels[1]

    token_predictions = np.argmax(tokens_logits, axis=-1)
    sequence_predictions = np.argmax(sequence_logits, axis=-1)

    cleaned_token_predictions = [
        token_predictions[i][token_labels[i] != -100] for i in range(len(token_predictions))
    ]
    cleaned_token_labels = [
        token_labels[i][token_labels[i] != -100] for i in range(len(token_labels))
    ]

    token_accuracies = [
        np.mean(cleaned_token_predictions[i] == cleaned_token_labels[i])
        for i in range(len(cleaned_token_predictions))
    ]
    token_accuracy = np.mean(token_accuracies)

    sequence_accuracy = np.mean(sequence_predictions == sequence_labels)

    transposed_logits = torch.tensor(tokens_logits).transpose(-2, -1)
    token_loss = F.cross_entropy(transposed_logits, torch.tensor(token_labels))
    sequence_loss = F.cross_entropy(torch.tensor(sequence_logits), torch.tensor(sequence_labels))

    return {
        "token_accuracy": token_accuracy,
        "sequence_accuracy": sequence_accuracy,
        "token_loss": token_loss,
        "sequence_loss": sequence_loss,
    }


def train(cfg):
    print("Config:")
    for k, v in cfg.items():
        print(f"{k}: {v}")
    model = BertForMultiTaskClassification(
        model_id=cfg.model_id,
        num_token_labels=cfg.num_token_labels,
        num_sequence_labels=cfg.num_sequence_labels,
        token_coef=cfg.token_coef,
        sequence_coef=cfg.sequence_coef,
        classifier_dropout=cfg.classifier_dropout,
    )

    train_dataset, val_dataset, test_dataset = prepare_data(cfg)

    run_name = f"{cfg.token_coef=}-{cfg.sequence_coef=}-{cfg.n_train}"
    args = TrainingArguments(
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        logging_strategy="steps",
        logging_steps=1,
        output_dir=cfg.output_dir + f"/{run_name}",
        seed=cfg.seed,
        run_name=run_name,
        report_to="wandb",
        label_names=["token_labels", "sequence_labels"],
        eval_strategy="epoch",
        eval_on_start=True,
        weight_decay=cfg.weight_decay,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(tokenizer),
        compute_metrics=compute_metrics,
    )

    # launch
    print("Training...")
    trainer.train()
    print("Training Done! 💥")
    print("Testing...")
    trainer.evaluate(test_dataset, metric_key_prefix="test")
    print("Testing Done! 💥")
    return


@hydra.main(config_path="./", config_name="config", version_base=None)
def main(cfg):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()
