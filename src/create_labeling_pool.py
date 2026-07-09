from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/cleaned_commits.csv")
LABELED_FILE = Path("data/labeled_commits.csv")
OUTPUT_FILE = Path("data/labeling_pool.csv")

POOL_SIZE = 400
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(INPUT_FILE)

    if LABELED_FILE.exists():
        labeled_df = pd.read_csv(LABELED_FILE)
    else:
        labeled_df = pd.DataFrame()

    print(f"Cleaned commits: {len(df)}")
    print(f"Already labeled: {len(labeled_df)}")

    return df, labeled_df


def add_candidate_reason(df, reason):
    result = df.copy()
    result["selection_reason"] = reason
    return result


def sample_candidates(df):
    candidates = []

    # Very short messages
    short_messages = df[
        df["message_word_count"] <= 3
    ]

    candidates.append(
        add_candidate_reason(
            short_messages.sample(
                n=min(70, len(short_messages)),
                random_state=RANDOM_STATE,
            ),
            "short_message",
        )
    )

    # Tiny commits
    tiny_commits = df[
        df["total_changes"] <= 5
    ]

    candidates.append(
        add_candidate_reason(
            tiny_commits.sample(
                n=min(60, len(tiny_commits)),
                random_state=RANDOM_STATE + 1,
            ),
            "tiny_commit",
        )
    )

    # Large commits
    large_threshold = df["total_changes"].quantile(0.90)

    large_commits = df[
        df["total_changes"] >= large_threshold
    ]

    candidates.append(
        add_candidate_reason(
            large_commits.sample(
                n=min(40, len(large_commits)),
                random_state=RANDOM_STATE + 2,
            ),
            "large_commit",
        )
    )

    # Source-code commits
    source_commits = df[
        df["source_files_changed"] > 0
    ]

    candidates.append(
        add_candidate_reason(
            source_commits.sample(
                n=min(60, len(source_commits)),
                random_state=RANDOM_STATE + 3,
            ),
            "source_code",
        )
    )

    # Test commits
    test_commits = df[
        df["test_files_changed"] > 0
    ]

    candidates.append(
        add_candidate_reason(
            test_commits.sample(
                n=min(40, len(test_commits)),
                random_state=RANDOM_STATE + 4,
            ),
            "test_change",
        )
    )

    # Documentation-only commits
    documentation_commits = df[
        (df["doc_files_changed"] > 0)
        & (df["source_files_changed"] == 0)
        & (df["test_files_changed"] == 0)
    ]

    candidates.append(
        add_candidate_reason(
            documentation_commits.sample(
                n=min(40, len(documentation_commits)),
                random_state=RANDOM_STATE + 5,
            ),
            "documentation_only",
        )
    )

    # Configuration-only commits
    config_commits = df[
        (df["config_files_changed"] > 0)
        & (df["source_files_changed"] == 0)
        & (df["test_files_changed"] == 0)
    ]

    candidates.append(
        add_candidate_reason(
            config_commits.sample(
                n=min(40, len(config_commits)),
                random_state=RANDOM_STATE + 6,
            ),
            "configuration_only",
        )
    )

    # Merge commits
    merge_commits = df[
        df["is_merge_commit"] == True
    ]

    candidates.append(
        add_candidate_reason(
            merge_commits.sample(
                n=min(30, len(merge_commits)),
                random_state=RANDOM_STATE + 7,
            ),
            "merge_commit",
        )
    )

    # Bot-authored commits
    bot_commits = df[
        df["author_login"]
        .astype(str)
        .str.contains(
            r"\[bot\]$",
            case=False,
            regex=True,
            na=False,
        )
    ]

    candidates.append(
        add_candidate_reason(
            bot_commits.sample(
                n=min(40, len(bot_commits)),
                random_state=RANDOM_STATE + 8,
            ),
            "bot_commit",
        )
    )

    # General random sample
    candidates.append(
        add_candidate_reason(
            df.sample(
                n=min(100, len(df)),
                random_state=RANDOM_STATE + 9,
            ),
            "random",
        )
    )

    candidate_df = pd.concat(
        candidates,
        ignore_index=True,
    )

    return candidate_df


def combine_duplicate_reasons(candidate_df):
    reason_map = (
        candidate_df
        .groupby(["repository", "sha"])["selection_reason"]
        .apply(
            lambda values: "|".join(
                sorted(set(values))
            )
        )
        .reset_index()
    )

    base_columns = [
        column
        for column in candidate_df.columns
        if column != "selection_reason"
    ]

    unique_candidates = (
        candidate_df[base_columns]
        .drop_duplicates(
            subset=["repository", "sha"]
        )
    )

    return unique_candidates.merge(
        reason_map,
        on=["repository", "sha"],
        how="left",
    )


def add_labeled_commits(pool, cleaned_df, labeled_df):
    if labeled_df.empty:
        return pool

    labeled_keys = labeled_df[
        ["repository", "sha"]
    ].drop_duplicates()

    existing_labeled = cleaned_df.merge(
        labeled_keys,
        on=["repository", "sha"],
        how="inner",
    )

    existing_labeled["selection_reason"] = (
        "already_labeled"
    )

    combined = pd.concat(
        [
            existing_labeled,
            pool,
        ],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=["repository", "sha"],
        keep="first",
    )

    return combined


def balance_repositories(pool, pool_size):
    repositories = sorted(
        pool["repository"].unique()
    )

    selected_parts = []

    per_repository = max(
        1,
        pool_size // len(repositories),
    )

    for index, repository in enumerate(repositories):
        repository_df = pool[
            pool["repository"] == repository
        ]

        selected = repository_df.sample(
            n=min(
                per_repository,
                len(repository_df),
            ),
            random_state=RANDOM_STATE + index,
        )

        selected_parts.append(selected)

    selected_pool = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    selected_pool = selected_pool.drop_duplicates(
        subset=["repository", "sha"]
    )

    return selected_pool


def fill_remaining_slots(selected_pool, candidate_pool, pool_size):
    if len(selected_pool) >= pool_size:
        return selected_pool.head(pool_size)

    selected_keys = set(
        zip(
            selected_pool["repository"],
            selected_pool["sha"],
        )
    )

    remaining = candidate_pool[
        ~candidate_pool.apply(
            lambda row: (
                row["repository"],
                row["sha"],
            ) in selected_keys,
            axis=1,
        )
    ]

    needed = pool_size - len(selected_pool)

    if needed > 0 and not remaining.empty:
        extra = remaining.sample(
            n=min(needed, len(remaining)),
            random_state=RANDOM_STATE + 100,
        )

        selected_pool = pd.concat(
            [selected_pool, extra],
            ignore_index=True,
        )

    return selected_pool


def ensure_labeled_commits_in_final_pool(
    final_pool,
    cleaned_df,
    labeled_df,
):
    if labeled_df.empty:
        return final_pool

    labeled_keys = labeled_df[
        ["repository", "sha"]
    ].drop_duplicates()

    labeled_rows = cleaned_df.merge(
        labeled_keys,
        on=["repository", "sha"],
        how="inner",
    )

    labeled_rows["selection_reason"] = "already_labeled"

    labeled_key_set = set(
        zip(
            labeled_rows["repository"],
            labeled_rows["sha"],
        )
    )

    unlabeled_pool_rows = final_pool[
        ~final_pool.apply(
            lambda row: (
                row["repository"],
                row["sha"],
            ) in labeled_key_set,
            axis=1,
        )
    ]

    slots_for_unlabeled = max(
        0,
        POOL_SIZE - len(labeled_rows),
    )

    unlabeled_pool_rows = unlabeled_pool_rows.head(
        slots_for_unlabeled
    )

    final_pool = pd.concat(
        [
            labeled_rows,
            unlabeled_pool_rows,
        ],
        ignore_index=True,
    )

    return final_pool.drop_duplicates(
        subset=["repository", "sha"]
    )


def save_pool(pool):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pool.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nSaved labeling pool: {OUTPUT_FILE}")
    print(f"Pool size: {len(pool)}")

    print("\nRepository distribution:")
    print(
        pool["repository"]
        .value_counts()
        .sort_index()
    )

    print("\nSelection reasons:")
    print(
        pool["selection_reason"]
        .str.get_dummies(sep="|")
        .sum()
        .sort_values(ascending=False)
    )


def main():
    cleaned_df, labeled_df = load_data()

    candidate_pool = sample_candidates(cleaned_df)

    candidate_pool = combine_duplicate_reasons(
        candidate_pool
    )

    print(
        f"Unique diverse candidates: "
        f"{len(candidate_pool)}"
    )

    candidate_pool = add_labeled_commits(
        candidate_pool,
        cleaned_df,
        labeled_df,
    )

    selected_pool = balance_repositories(
        candidate_pool,
        POOL_SIZE,
    )

    selected_pool = fill_remaining_slots(
        selected_pool,
        candidate_pool,
        POOL_SIZE,
    )

    final_pool = ensure_labeled_commits_in_final_pool(
        selected_pool,
        cleaned_df,
        labeled_df,
    )

    save_pool(final_pool)


if __name__ == "__main__":
    main()