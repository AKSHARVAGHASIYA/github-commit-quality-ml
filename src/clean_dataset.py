from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/raw_commits.csv")
OUTPUT_FILE = Path("data/cleaned_commits.csv")
REPORT_FILE = Path("reports/cleaning_report.txt")


def load_dataset():
    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} rows.")

    return df


def remove_duplicates(df):
    before = len(df)

    df = df.drop_duplicates(
        subset=["repository", "sha"]
    ).copy()

    removed = before - len(df)

    print(f"Removed duplicate commits: {removed}")

    return df, removed


def clean_author_login(df):
    missing_before = df["author_login"].isna().sum()

    df["author_login"] = (
        df["author_login"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )

    print(
        f"Missing author_login values replaced: "
        f"{missing_before}"
    )

    return df, missing_before


def clean_commit_messages(df):
    missing_messages = df["commit_message"].isna().sum()

    df["commit_message"] = (
        df["commit_message"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    print(
        f"Missing commit messages replaced: "
        f"{missing_messages}"
    )

    return df, missing_messages


def clean_timing_feature(df):
    invalid_timing = (
        df["hours_since_previous_commit"] < 0
    ).sum()

    df.loc[
        df["hours_since_previous_commit"] < 0,
        "hours_since_previous_commit"
    ] = pd.NA

    print(
        f"Invalid timing values converted to missing: "
        f"{invalid_timing}"
    )

    return df, invalid_timing


def validate_numeric_columns(df):
    numeric_columns = [
        "files_changed",
        "additions",
        "deletions",
        "total_changes",
        "source_files_changed",
        "test_files_changed",
        "doc_files_changed",
        "config_files_changed",
        "message_length",
        "message_word_count",
    ]

    invalid_counts = {}

    for column in numeric_columns:
        invalid_count = (df[column] < 0).sum()

        invalid_counts[column] = int(invalid_count)

        if invalid_count > 0:
            df.loc[df[column] < 0, column] = pd.NA

    print("Numeric validation completed.")

    return df, invalid_counts


def validate_change_consistency(df):
    inconsistent = (
        df["total_changes"]
        != df["additions"] + df["deletions"]
    )

    count = inconsistent.sum()

    print(f"Inconsistent total_changes rows: {count}")

    return count


def save_cleaned_dataset(df):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Saved {len(df)} cleaned rows "
        f"to {OUTPUT_FILE}"
    )


def save_cleaning_report(
    original_rows,
    final_rows,
    duplicates_removed,
    missing_authors,
    missing_messages,
    invalid_timing,
    invalid_numeric_counts,
    inconsistent_changes,
):
    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(REPORT_FILE, "w") as file:
        file.write("DATASET CLEANING REPORT\n")
        file.write("=" * 50 + "\n\n")

        file.write(
            f"Original rows: {original_rows}\n"
        )

        file.write(
            f"Final rows: {final_rows}\n"
        )

        file.write(
            f"Duplicate commits removed: "
            f"{duplicates_removed}\n"
        )

        file.write(
            f"Missing authors replaced: "
            f"{missing_authors}\n"
        )

        file.write(
            f"Missing messages replaced: "
            f"{missing_messages}\n"
        )

        file.write(
            f"Invalid timing values converted to missing: "
            f"{invalid_timing}\n"
        )

        file.write(
            f"Inconsistent total_changes rows: "
            f"{inconsistent_changes}\n\n"
        )

        file.write("INVALID NUMERIC VALUES\n")

        for column, count in invalid_numeric_counts.items():
            file.write(
                f"{column}: {count}\n"
            )

    print(f"Cleaning report saved to {REPORT_FILE}")


def main():
    df = load_dataset()

    original_rows = len(df)

    df, duplicates_removed = remove_duplicates(df)

    df, missing_authors = clean_author_login(df)

    df, missing_messages = clean_commit_messages(df)

    df, invalid_timing = clean_timing_feature(df)

    df, invalid_numeric_counts = validate_numeric_columns(df)

    inconsistent_changes = validate_change_consistency(df)

    save_cleaned_dataset(df)

    save_cleaning_report(
        original_rows=original_rows,
        final_rows=len(df),
        duplicates_removed=duplicates_removed,
        missing_authors=missing_authors,
        missing_messages=missing_messages,
        invalid_timing=invalid_timing,
        invalid_numeric_counts=invalid_numeric_counts,
        inconsistent_changes=inconsistent_changes,
    )


if __name__ == "__main__":
    main()