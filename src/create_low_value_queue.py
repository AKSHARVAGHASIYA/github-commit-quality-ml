from pathlib import Path

import pandas as pd


POOL_FILE = Path("data/enriched_labeling_pool.csv")
LABELED_FILE = Path("data/labeled_commits.csv")
OUTPUT_FILE = Path("data/low_value_queue.csv")

QUEUE_SIZE = 100

MAX_PER_REPOSITORY = 15
MAX_PER_MESSAGE = 5

def load_data():
    pool_df = pd.read_csv(POOL_FILE)

    if LABELED_FILE.exists():
        labeled_df = pd.read_csv(LABELED_FILE)
    else:
        labeled_df = pd.DataFrame()

    return pool_df, labeled_df


def remove_already_reviewed(pool_df, labeled_df):
    """
    Remove every manually reviewed commit regardless of whether its label is
    USEFUL, UNCERTAIN, or LOW_VALUE.
    """
    if labeled_df.empty:
        return pool_df.copy()

    reviewed_keys = set(
        zip(
            labeled_df["repository"].astype(str),
            labeled_df["sha"].astype(str),
        )
    )

    keys = list(
        zip(
            pool_df["repository"].astype(str),
            pool_df["sha"].astype(str),
        )
    )

    mask = [
        key not in reviewed_keys
        for key in keys
    ]

    return pool_df.loc[mask].copy()


def numeric_series(df, column):
    if column not in df.columns:
        return pd.Series(0, index=df.index, dtype="float64")

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(0)


def text_series(df, column):
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="object")

    return df[column].fillna("").astype(str)


def calculate_candidate_score(df):
    """
    Rank commits that are plausible LOW_VALUE candidates.

    Stronger evidence comes from enriched diff information:
    changed filenames, patch text, lockfiles, generated files, and whether
    actual source/test code was changed.

    The score is only for queue prioritization. It is not an automatic label.
    """
    df = df.copy()

    message = text_series(df, "commit_message").str.lower().str.strip()
    filenames = text_series(df, "changed_filenames").str.lower()
    patches = text_series(df, "patches").str.lower()

    total_changes = numeric_series(df, "total_changes")
    files_changed = numeric_series(df, "files_changed")
    message_word_count = numeric_series(df, "message_word_count")

    source_files = numeric_series(df, "source_files_changed")
    test_files = numeric_series(df, "test_files_changed")
    doc_files = numeric_series(df, "doc_files_changed")
    config_files = numeric_series(df, "config_files_changed")

    lockfiles = numeric_series(df, "lockfile_count")
    generated_files = numeric_series(df, "generated_file_count")
    api_files = numeric_series(df, "api_files_count")
    files_with_patches = numeric_series(df, "patch_available_count")

    df["candidate_score"] = 0
    df["candidate_reasons"] = ""

    def add_score(mask, score, reason):
        df.loc[mask, "candidate_score"] += score

        current = df.loc[mask, "candidate_reasons"]

        df.loc[mask, "candidate_reasons"] = current.where(
            current == "",
            current + " | ",
        ) + reason

    # ---------------------------------------------------------
    # 1. Strong enriched diff evidence
    # ---------------------------------------------------------

    lockfile_only = (
        (lockfiles > 0)
        & (source_files == 0)
        & (test_files == 0)
        & (doc_files == 0)
    )

    add_score(
        lockfile_only,
        8,
        "lockfile-only change",
    )

    generated_only = (
        (generated_files > 0)
        & (source_files == 0)
        & (test_files == 0)
    )

    add_score(
        generated_only,
        7,
        "generated-files-only change",
    )

    whitespace_patch = patches.str.contains(
        r"whitespace|trailing space|indentation|reformat",
        regex=True,
        na=False,
    )

    add_score(
        whitespace_patch,
        4,
        "patch suggests formatting/whitespace change",
    )

    metadata_filename = filenames.str.contains(
        r"\.gitignore|\.gitattributes|"
        r"\.editorconfig|\.prettierignore|"
        r"\.eslintignore",
        regex=True,
        na=False,
    )

    metadata_only = (
        metadata_filename
        & (files_changed <= 2)
        & (source_files == 0)
        & (test_files == 0)
    )

    add_score(
        metadata_only,
        5,
        "small repository-metadata change",
    )

    # ---------------------------------------------------------
    # 2. Small non-code changes
    # ---------------------------------------------------------

    tiny_non_code = (
        (total_changes <= 5)
        & (source_files == 0)
        & (test_files == 0)
        & (api_files == 0)
    )

    add_score(
        tiny_non_code,
        5,
        "tiny change with no source/test/API files",
    )

    docs_only_small = (
        (doc_files > 0)
        & (source_files == 0)
        & (test_files == 0)
        & (config_files == 0)
        & (total_changes <= 10)
    )

    add_score(
        docs_only_small,
        4,
        "small documentation-only change",
    )

    config_only_small = (
        (config_files > 0)
        & (source_files == 0)
        & (test_files == 0)
        & (doc_files == 0)
        & (total_changes <= 10)
    )

    add_score(
        config_only_small,
        4,
        "small configuration-only change",
    )

    # ---------------------------------------------------------
    # 3. Diff availability and size evidence
    # ---------------------------------------------------------

    tiny_patch_commit = (
        (files_with_patches > 0)
        & (files_changed <= 2)
        & (total_changes <= 4)
    )

    add_score(
        tiny_patch_commit,
        3,
        "tiny commit with inspectable patch",
    )

    single_file_tiny = (
        (files_changed == 1)
        & (total_changes <= 3)
    )

    add_score(
        single_file_tiny,
        3,
        "single-file tiny change",
    )

    # ---------------------------------------------------------
    # 4. Commit-message evidence
    # ---------------------------------------------------------

    exact_low_information_message = message.str.match(
        r"^(fix|fixed|update|updated|change|changed|"
        r"typo|format|formatting|cleanup|clean up|"
        r"wip|test|testing|misc|minor|small change|"
        r"remove|removed|rename|renamed|bump)$",
        na=False,
    )

    add_score(
        exact_low_information_message,
        4,
        "low-information commit message",
    )

    explicit_low_value_message = message.str.contains(
        r"\bwip\b|\btypo\b|\bwhitespace\b|"
        r"\bformatting\b|\breformat\b|"
        r"\bminor cleanup\b|\bversion bump\b|"
        r"\bdependency bump\b|\bbump version\b",
        regex=True,
        na=False,
    )

    add_score(
        explicit_low_value_message,
        3,
        "message suggests mechanical/minor change",
    )

    very_short_message = message_word_count <= 3

    add_score(
        very_short_message,
        1,
        "very short commit message",
    )

    # ---------------------------------------------------------
    # 5. Penalize commits with stronger engineering evidence
    # ---------------------------------------------------------

    substantial_source_change = (
        (source_files > 0)
        & (total_changes >= 20)
    )

    add_score(
        substantial_source_change,
        -5,
        "substantial source-code change",
    )

    test_and_source_change = (
        (source_files > 0)
        & (test_files > 0)
    )

    add_score(
        test_and_source_change,
        -4,
        "source and tests changed together",
    )

    large_patch = total_changes >= 100

    add_score(
        large_patch,
        -4,
        "large change",
    )

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

    # Normalize messages so repeated commit patterns
    # cannot dominate the manual-labeling queue.
    candidates["normalized_message"] = (
        candidates["commit_message"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Cap repeated commit-message patterns.
    candidates = (
        candidates
        .groupby(
            "normalized_message",
            group_keys=False,
            sort=False,
        )
        .head(MAX_PER_MESSAGE)
    )

    # Cap candidates from any single repository.
    candidates = (
        candidates
        .groupby(
            "repository",
            group_keys=False,
            sort=False,
        )
        .head(MAX_PER_REPOSITORY)
    )

    # Restore global candidate priority.
    candidates = candidates.sort_values(
        by=[
            "candidate_score",
            "total_changes",
            "message_word_count",
        ],
        ascending=[False, True, True],
    )

    candidates = candidates.head(QUEUE_SIZE)

    # Internal queue-generation column is not needed
    # during manual labeling.
    candidates = candidates.drop(
        columns=["normalized_message"]
    )

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
        "candidate_reasons",
    ]

    print(
        queue_df[columns]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nQueue saved to {OUTPUT_FILE}")


def main():
    pool_df, labeled_df = load_data()

    print(f"Loaded enriched pool commits: {len(pool_df)}")
    print(f"Loaded reviewed commits: {len(labeled_df)}")

    unlabeled_df = remove_already_reviewed(
        pool_df,
        labeled_df,
    )

    print(
        f"Remaining unreviewed pool commits: "
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
