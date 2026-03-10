from datasets import load_dataset
import pandas as pd
import os


DATA_DIR = "../data"


def load_wili_dataset(val_split=0.1):
    """
    Load WiLI-2018 dataset, create validation split,
    save both Arrow format and CSV format.
    """

    print("Loading WiLI-2018 dataset...")

    os.makedirs(DATA_DIR, exist_ok=True)

    dataset = load_dataset(
        "wili_2018",
        cache_dir=DATA_DIR
    )

    train_data = dataset["train"]
    test_data = dataset["test"]

    # Create validation split
    split = train_data.train_test_split(test_size=val_split, seed=42)

    train_split = split["train"]
    val_split = split["test"]

    print(f"Train samples: {len(train_split)}")
    print(f"Validation samples: {len(val_split)}")
    print(f"Test samples: {len(test_data)}")

    # -------------------------
    # SAVE ARROW DATASETS
    # -------------------------

    train_split.save_to_disk(f"{DATA_DIR}/train_arrow")
    val_split.save_to_disk(f"{DATA_DIR}/validation_arrow")
    test_data.save_to_disk(f"{DATA_DIR}/test_arrow")

    print("Arrow datasets saved.")

    # -------------------------
    # SAVE CSV DATASETS
    # -------------------------

    train_df = train_split.to_pandas()
    val_df = val_split.to_pandas()
    test_df = test_data.to_pandas()

    train_df.to_csv(f"{DATA_DIR}/train.csv", index=False)
    val_df.to_csv(f"{DATA_DIR}/validation.csv", index=False)
    test_df.to_csv(f"{DATA_DIR}/test.csv", index=False)

    print("CSV datasets saved.")

    num_languages = len(set(train_split["label"]))

    return train_split, val_split, test_data, num_languages


if __name__ == "__main__":

    train, val, test, num_languages = load_wili_dataset()

    print("\nExample sample:")
    print(train[0])

    print("\nDataset features:")
    print(train.features)