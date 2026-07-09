from pathlib import Path

import pandas as pd


POOL_FILE = Path("data/labeling_pool.csv")
LABELED_FILE = Path("data/labeled_commits.csv")
OUTPUT_FILE = Path("data/low_value_queue.csv")

QUEUE_SIZE = 100


def load_data():
    pool_df = pd.read_csv(POOL_FILE)

    if LABELED_FILE.exists():
        labeled_df = pd.read_csv(LABELED_FILE)
    else:
        labeled_df = pd.DataFrame()

    return pool_df, labeled_df


def remove_already_labeled(pool_df, labeled_df):
    if labeled_df.empty:
        return pool_df.copy()

    labeled_keys = set(
        zip(
            labeled_df["repository"].astype(str),
            labeled_df["sha"].astype(str),
        )
    )

    mask = pool_df.apply(
        lambda row: (
            str(row["repository"]),
            str(row["sha"]),
        ) not in labeled_keys,
        axis=1,
    )

    return pool_df[mask].copy()


def calculate_candidate_score(df):
    df = df.copy()

    message = df["commit_message"].fillna("").astype(str)
    message_lower = message.str.lower()

    df["candidate_score"] = 0

    # Very short commit messages.
    df.loc[
        df["message_word_count"] <= 3,
        "candidate_score",
    ] += 3

    # Tiny commits.
    df.loc[
        df["total_changes"] <= 2,
        "candidate_score",
    ] += 3

    df.loc[
        df["files_changed"] == 1,
        "candidate_score",
    ] += 1

    # Common low-information messages.
    low_information_patterns = (
        r"^(fix|fixed|update|updated|change|changed|"
        r"typo|format|formatting|cleanup|clean up|"
        r"wip|test|testing|misc|minor|small change|"
        r"remove|removed|rename|renamed|bump)$"
    )

    df.loc[
        message_lower.str.strip().str.match(
            low_information_patterns,
            na=False,
        ),
        "candidate_score",
    ] += 5

    # Explicit WIP / typo / formatting indicators.
    df.loc[
        message_lower.str.contains(
            r"\bwip\b|\btypo\b|\bformatting\b|"
            r"\bwhitespace\b|\bminor cleanup\b",
            regex=True,
            na=False,
        ),
        "candidate_score",
    ] += 3

    # Documentation-only tiny changes.
    documentation_only = (
        (df["doc_files_changed"] > 0)
        & (df["source_files_changed"] == 0)
        & (df["test_files_changed"] == 0)
        & (df["config_files_changed"] == 0)
        & (df["total_changes"] <= 5)
    )

    df.loc[
        documentation_only,
        "candidate_score",
    ] += 2

    # Configuration-only tiny changes.
    configuration_only = (
        (df["config_files_changed"] > 0)
        & (df["source_files_changed"] == 0)
        & (df["test_files_changed"] == 0)
        & (df["doc_files_changed"] == 0)
        & (df["total_changes"] <= 5)
    )

    df.loc[
        configuration_only,
        "candidate_score",
    ] += 2

    return df


def create_queue(df):
    candidates = df[
        df["candidate_score"] > 0
    ].copy()

    candidates = candidates.sort_values(
        by=[
            "candidate_score",
            "total_changes",
            "message_word_count",
        ],
        ascending=[False, True, True],
    )

    candidates = candidates.head(QUEUE_SIZE)

    return candidates


def save_queue(queue_df):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    queue_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


def print_summary(queue_df):
    print("\nLOW-VALUE CANDIDATE QUEUE")
    print("=" * 70)

    print(f"Queue size: {len(queue_df)}")

    if queue_df.empty:
        print("No candidates found.")
        return

    print("\nCandidate score distribution:")
    print(
        queue_df["candidate_score"]
        .value_counts()
        .sort_index(ascending=False)
    )

    print("\nTop 10 candidates:")

    columns = [
        "repository",
        "sha",
        "commit_message",
        "total_changes",
        "candidate_score",
    ]

    print(
        queue_df[columns]
        .head(10)
        .to_string(index=False)
    )

    print(
        f"\nQueue saved to {OUTPUT_FILE}"
    )


def main():
    pool_df, labeled_df = load_data()

    print(f"Loaded pool commits: {len(pool_df)}")
    print(f"Loaded labeled commits: {len(labeled_df)}")

    unlabeled_df = remove_already_labeled(
        pool_df,
        labeled_df,
    )

    print(
        f"Remaining unlabeled pool commits: "
        f"{len(unlabeled_df)}"
    )

    scored_df = calculate_candidate_score(
        unlabeled_df
    )

    queue_df = create_queue(scored_df)

    save_queue(queue_df)

    print_summary(queue_df)


if __name__ == "__main__":
    main()