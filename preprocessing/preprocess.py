import os
from dataclasses import dataclass
from typing import Literal, Optional

from datasets import Dataset, DatasetDict, load_dataset
from openai import OpenAI
from pydantic import BaseModel


@dataclass
class Config:
    n_train: int = 1000
    n_val: int = 250
    n_test: int = 250
    seed: int = 0
    plain_output_dir: str = "../data/conll2003-plain"
    augmented_output_dir: str = "../data/conll2003-augmented"
    dataset_name: str = "eriktks/conll2003"
    target_column: str = "ner_tags"
    gpt_model: str = "gpt-4o-mini"


def get_plain_dataset(cfg: Config) -> Dataset:
    dataset = load_dataset(cfg.dataset_name, keep_in_memory=True, trust_remote_code=True)
    column_names = dataset["train"].column_names
    column_names_to_remove = [
        key for key in column_names if key not in (cfg.target_column, "tokens", "id")
    ]
    dataset = dataset.shuffle(seed=cfg.seed)
    train_dataset = (
        dataset["train"].select(range(cfg.n_train)).remove_columns(column_names_to_remove)
    )
    val_dataset = (
        dataset["validation"].select(range(cfg.n_val)).remove_columns(column_names_to_remove)
    )
    test_dataset = dataset["test"].select(range(cfg.n_test)).remove_columns(column_names_to_remove)
    new_dataset = DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )
    return new_dataset


class DocumentLevelLabel(BaseModel):
    topic: Literal["World", "Sports", "Business", "Technology", "Others"]


def topic_to_int(topic: str) -> int:
    categories = ["World", "Sports", "Business", "Technology", "Others"]
    return categories.index(topic)


def augment_example(
    example: dict[str, any], cfg: Optional[Config] = None, client: Optional[OpenAI] = None
) -> dict[str, any]:
    tokens = example["tokens"]
    text = " ".join(tokens)
    categories = ["World", "Sports", "Business", "Technology", "Others"]
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant whose role is to classify the following text into one of the following categories: {', '.join(categories)}. Only respond with the category name.",
        },
        {"role": "user", "content": text},
    ]
    completion = client.beta.chat.completions.parse(
        model=cfg.gpt_model, messages=messages, response_format=DocumentLevelLabel
    )

    response = completion.choices[0].message

    print("-" * 80)
    print("Text: ", text)
    print("Response: ", response.parsed)

    if response.parsed:
        example["topic"] = topic_to_int(response.parsed.topic)
    elif response.refusal:
        print("Refused to classify the text")
        print("Messages: ", messages)
        print("Response: ", response)
        raise RuntimeError("Refused to classify the text")
    else:
        print("Failed to classify the text")
        print("Messages: ", messages)
        print("Response: ", response)
        raise RuntimeError("Failed to classify the text")

    return example


def augment_dataset(cfg: Config, dataset: Dataset) -> Dataset:
    client = OpenAI()
    print("Augmenting dataset...")
    augmented_dataset = dataset.map(augment_example, fn_kwargs={"cfg": cfg, "client": client})
    return augmented_dataset


def main():
    cfg = Config()
    print("Config: ", cfg.__dict__)
    dataset = get_plain_dataset(cfg)
    print("Plain dataset: ", dataset)
    os.makedirs(cfg.plain_output_dir, exist_ok=True)
    dataset.save_to_disk(cfg.plain_output_dir)
    print("Sample from plain dataset: ", dataset["train"][0])

    augmented_dataset = augment_dataset(cfg, dataset)
    print("Augmented dataset: ", augmented_dataset)
    os.makedirs(cfg.augmented_output_dir, exist_ok=True)
    augmented_dataset.save_to_disk(cfg.augmented_output_dir)
    print("Sample from augmented dataset: ", augmented_dataset["train"][0])


if __name__ == "__main__":
    main()
