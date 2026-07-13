from pathlib import Path

import pandas as pd

from src.github_collector import classify_file


INPUT_FILE = Path("data/enriched_labeling_pool.csv")

CATEGORY_COLUMNS = [
    "source_files_changed",
    "test_files_changed",
    "doc_files_changed",
    "config_files_changed",
]

CATEGORY_TO_COLUMN = {
    "source": "source_files_changed",
    "test": "test_files_changed",
    "documentation": "doc_files_changed",
    "config": "config_files_changed",
}


def recompute_category_counts(changed_filenames):
    counts = {column: 0 for column in CATEGORY_COLUMNS}

    if pd.isna(changed_filenames):
        return counts

    for filename in str(changed_filenames).splitlines():
        filename = filename.strip()

        if not filename:
            continue

        category = classify_file(filename)
        column = CATEGORY_TO_COLUMN.get(category)

        if column is not None:
            counts[column] += 1

    return counts


def main():
    dataframe = pd.read_csv(INPUT_FILE)

    missing_columns = [
        column
        for column in ["changed_filenames", *CATEGORY_COLUMNS]
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    repaired_counts = dataframe["changed_filenames"].apply(
        recompute_category_counts
    ).apply(pd.Series)

    old_counts = (
        dataframe[CATEGORY_COLUMNS]
        .fillna(0)
        .astype(int)
        .reset_index(drop=True)
    )

    repaired_counts = (
        repaired_counts[CATEGORY_COLUMNS]
        .astype(int)
        .reset_index(drop=True)
    )

    mismatch_mask = old_counts.ne(repaired_counts).any(axis=1)
    rows_repaired = int(mismatch_mask.sum())

    dataframe.loc[:, CATEGORY_COLUMNS] = repaired_counts.to_numpy()

    dataframe.to_csv(INPUT_FILE, index=False)

    print(f"Loaded rows: {len(dataframe)}")
    print(f"Rows repaired: {rows_repaired}")
    print(f"Rows unchanged: {len(dataframe) - rows_repaired}")
    print(f"Repaired file: {INPUT_FILE}")


if __name__ == "__main__":
    main()