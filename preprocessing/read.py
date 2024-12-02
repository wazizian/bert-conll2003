from datasets import Dataset, DatasetDict, load_from_disk

ner_tags = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}
list_ner_tags = list(ner_tags)

categories = ["World", "Sports", "Business", "Technology", "Others"]


def int_to_topic(i: int) -> str:
    return categories[i]


def int_to_ner_tag(i: int) -> str:
    return list_ner_tags[i]


def read_dataset(data_dir: str) -> Dataset:
    dataset = load_from_disk(data_dir)
    for split, d in dataset.items():
        print("\n" + "=" * 80)
        print(f"{split} dataset of size {len(d)}")
        d = d.to_iterable_dataset().shuffle()
        for example in d:
            print("\n" + "-" * 80)
            text = " ".join(example["tokens"])
            print(f"Topic: ", int_to_topic(example["topic"]))
            print(f"Text: ", text)
            print(f"Labels: ", [int_to_ner_tag(i) for i in example["ner_tags"]])


if __name__ == "__main__":
    read_dataset("../data/conll2003-augmented")
