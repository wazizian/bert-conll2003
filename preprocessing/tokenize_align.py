from dataclasses import dataclass

from datasets import DatasetDict
from transformers import AutoTokenizer


@dataclass
class Config:
    seed: int = 0
    input_dir: str = "../data/conll2003-augmented"
    output_dir: str = "../data/conll2003-tokenized"
    model_id: str = "bert-base-cased"


def get_data(cfg: Config) -> DatasetDict:
    dataset = DatasetDict.load_from_disk(cfg.input_dir)
    return dataset


# The token alignement procedure is taken from
# https://huggingface.co/learn/nlp-course/chapter7/2#fine-tuning-the-model-with-keras
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize(cfg: Config, dataset: DatasetDict):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    def tokenize_examples(examples):
        b = len(examples["tokens"])
        tokenized = tokenizer(
            examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True
        )
        tokenized["token_labels"] = []
        for i in range(b):
            tokenized["token_labels"].append(
                align_labels_with_tokens(examples["ner_tags"][i], tokenized.word_ids(i))
            )
        tokenized["sequence_labels"] = examples["topic"]
        return tokenized

    tokenized_dataset = dataset.map(tokenize_examples, batched=True).remove_columns(
        dataset["train"].column_names
    )
    return tokenized_dataset


if __name__ == "__main__":
    cfg = Config()
    dataset = get_data(cfg)
    print("Dataset:", dataset)
    tokenized_dataset = tokenize(cfg, dataset)
    print("Tokenized dataset:", tokenized_dataset)
    tokenized_dataset.save_to_disk(cfg.output_dir)
