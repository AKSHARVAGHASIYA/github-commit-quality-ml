from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/raw_commits.csv")


def load_dataset():
    df = pd.read_csv(INPUT_FILE)

    print("\n========== DATASET LOADED ==========")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df


def show_columns(df):
    print("\n========== COLUMNS ==========")

    for column in df.columns:
        print(column)


def check_missing_values(df):
    print("\n========== MISSING VALUES ==========")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)


def check_duplicates(df):
    print("\n========== DUPLICATES ==========")

    duplicate_count = df.duplicated(
        subset=["repository", "sha"]
    ).sum()

    print("Duplicate repository + SHA rows:", duplicate_count)


def check_repository_distribution(df):
    print("\n========== REPOSITORY DISTRIBUTION ==========")

    print(
        df["repository"]
        .value_counts()
        .sort_index()
    )


def check_numeric_statistics(df):
    print("\n========== NUMERIC STATISTICS ==========")

    columns = [
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
        "hours_since_previous_commit",
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    print(
        df[available_columns]
        .describe()
        .T
    )


def check_merge_commits(df):
    print("\n========== MERGE COMMITS ==========")

    if "is_merge_commit" not in df.columns:
        print("Column is_merge_commit does not exist.")
        return

    print(df["is_merge_commit"].value_counts(dropna=False))

    merge_percentage = df["is_merge_commit"].mean() * 100

    print(f"Merge commit percentage: {merge_percentage:.2f}%")


def check_zero_change_commits(df):
    print("\n========== ZERO CHANGE COMMITS ==========")

    zero_change_commits = df[df["total_changes"] == 0]

    print("Zero-change commits:", len(zero_change_commits))

    if not zero_change_commits.empty:
        print(
            zero_change_commits[
                [
                    "repository",
                    "sha",
                    "commit_message",
                    "is_merge_commit",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )


def check_short_messages(df):
    print("\n========== SHORTEST COMMIT MESSAGES ==========")

    columns = [
        "repository",
        "sha",
        "commit_message",
        "message_length",
        "total_changes",
    ]

    print(
        df[columns]
        .sort_values("message_length")
        .head(20)
        .to_string(index=False)
    )


def check_large_commits(df):
    print("\n========== LARGEST COMMITS ==========")

    columns = [
        "repository",
        "sha",
        "commit_message",
        "files_changed",
        "additions",
        "deletions",
        "total_changes",
    ]

    print(
        df[columns]
        .sort_values(
            "total_changes",
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )


def main():
    df = load_dataset()

    show_columns(df)
    check_missing_values(df)
    check_duplicates(df)
    check_repository_distribution(df)
    check_numeric_statistics(df)
    check_merge_commits(df)
    check_zero_change_commits(df)
    check_short_messages(df)
    check_large_commits(df)


if __name__ == "__main__":
    main()