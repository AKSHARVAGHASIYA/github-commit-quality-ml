from pathlib import Path

import pandas as pd
import argparse


POOL_FILE = Path("data/labeling_pool.csv")
LABELED_FILE = Path("data/labeled_commits.csv")
LOW_VALUE_QUEUE_FILE = Path("data/low_value_queue.csv")
ENRICHED_POOL_FILE = Path("data/enriched_labeling_pool.csv")

MAX_PATCH_PREVIEW_CHARS = 3500
MAX_FILE_SUMMARY_CHARS = 2000

VALID_LABELS = {
    "u": "USEFUL",
    "l": "LOW_VALUE",
    "?": "UNCERTAIN",
}


def load_pool(queue_name=None):
    if queue_name == "low-value":
        pool_file = LOW_VALUE_QUEUE_FILE

        print(
            "Loaded low-value candidate queue:",
            end=" ",
        )

    else:
        pool_file = ENRICHED_POOL_FILE

        print(
            "Loaded enriched labeling pool:",
            end=" ",
        )

    if not pool_file.exists():
        raise FileNotFoundError(
            f"Pool file not found: {pool_file}"
        )

    df = pd.read_csv(pool_file)

    print(f"{len(df)} commits.")

    return df


def load_existing_labels():
    if LABELED_FILE.exists():
        labeled_df = pd.read_csv(LABELED_FILE)

        print(
            f"Loaded existing labels: "
            f"{len(labeled_df)} commits."
        )

        return labeled_df

    print("No existing labeled dataset found.")

    return pd.DataFrame()


def get_commit_key(row):
    return (
        str(row["repository"]),
        str(row["sha"]),
    )


def get_labeled_keys(labeled_df):
    if labeled_df.empty:
        return set()

    return set(
        zip(
            labeled_df["repository"].astype(str),
            labeled_df["sha"].astype(str),
        )
    )


def save_label(row, label, labeled_df):
    row_data = row.to_dict()

    row_data["quality_label"] = label

    new_row = pd.DataFrame([row_data])

    if labeled_df.empty:
        updated_df = new_row

    else:
        updated_df = pd.concat(
            [labeled_df, new_row],
            ignore_index=True,
        )

    updated_df = updated_df.drop_duplicates(
        subset=["repository", "sha"],
        keep="last",
    )

    LABELED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    updated_df.to_csv(
        LABELED_FILE,
        index=False,
    )

    return updated_df


def print_label_distribution(labeled_df):
    print("\nCURRENT LABEL DISTRIBUTION")
    print("-" * 75)

    if labeled_df.empty:
        print("No labels assigned yet.")
        return

    print(
        labeled_df["quality_label"]
        .value_counts()
    )


def print_commit(row, pool_position, pool_size, reviewed_count):
    print("\n" + "=" * 75)

    print(
        f"POOL COMMIT {pool_position} / {pool_size}"
    )

    print(
        f"REVIEWED {reviewed_count} / {pool_size}"
    )

    print("=" * 75)

    print(f"Repository:       {row['repository']}")
    print(f"SHA:              {row['sha']}")
    print(f"Author:           {row['author_login']}")
    print(f"Date:             {row['commit_date']}")

    print(
        f"Selection reason: "
        f"{row.get('selection_reason', 'unknown')}"
    )

    print("\nCOMMIT MESSAGE")
    print("-" * 75)

    print(row["commit_message"])

    print("\nCHANGE STATISTICS")
    print("-" * 75)

    print(f"Files changed:    {row['files_changed']}")
    print(f"Additions:        {row['additions']}")
    print(f"Deletions:        {row['deletions']}")
    print(f"Total changes:    {row['total_changes']}")

    print("\nFILE CATEGORIES")
    print("-" * 75)

    print(
        f"Source files:     "
        f"{row['source_files_changed']}"
    )

    print(
        f"Test files:       "
        f"{row['test_files_changed']}"
    )

    print(
        f"Documentation:    "
        f"{row['doc_files_changed']}"
    )

    print(
        f"Configuration:    "
        f"{row['config_files_changed']}"
    )

    print("\nOTHER INFORMATION")
    print("-" * 75)

    print(
        f"Merge commit:     "
        f"{row['is_merge_commit']}"
    )

    timing = row["hours_since_previous_commit"]

    if pd.isna(timing):
        timing = "Unknown"

    print(
        f"Hours since previous commit: "
        f"{timing}"
    )
    
    if ( "file_summary" in row.index or "patch_summary" in row.index ):
        display_enriched_evidence(row)

    print("\nLABEL OPTIONS")
    print("[u] USEFUL")
    print("[l] LOW_VALUE")
    print("[?] UNCERTAIN")
    print("[s] SKIP FOR NOW")
    print("[q] SAVE AND QUIT")

def safe_text(value):
    if pd.isna(value):
        return ""

    return str(value)


def display_enriched_evidence(row):
    print("\nCHANGED FILES")
    print("-" * 75)

    file_summary = safe_text(
        row.get("file_summary", "")
    )

    if file_summary:
        print(file_summary[:MAX_FILE_SUMMARY_CHARS])

        if len(file_summary) > MAX_FILE_SUMMARY_CHARS:
            print("\n[File summary truncated]")

    else:
        print("No file summary available.")

    print("\nPATCH PREVIEW")
    print("-" * 75)

    patch_summary = safe_text(
        row.get("patch_summary", "")
    )

    if patch_summary:
        print(patch_summary[:MAX_PATCH_PREVIEW_CHARS])

        if len(patch_summary) > MAX_PATCH_PREVIEW_CHARS:
            print("\n[Patch preview truncated]")

    else:
        print("No patch data available.")

    print("\nENRICHMENT SUMMARY")
    print("-" * 75)

    print(
        f"API files count:         "
        f"{row.get('api_files_count', 'Unknown')}"
    )

    print(
        f"Files with patches:      "
        f"{row.get('patch_available_count', 'Unknown')}"
    )

    print(
        f"Lockfiles changed:       "
        f"{row.get('lockfile_count', 'Unknown')}"
    )

    print(
        f"Generated files changed: "
        f"{row.get('generated_file_count', 'Unknown')}"
    )



def main():
    parser = argparse.ArgumentParser(
        description="Interactive GitHub commit labeling tool."
    )

    parser.add_argument(
        "--queue",
        choices=["low-value"],
        default=None,
        help="Choose a specialized labeling queue.",
    )

    args = parser.parse_args()

    pool_df = load_pool(args.queue)

    labeled_df = load_existing_labels()

    labeled_keys = get_labeled_keys(labeled_df)

    pool_keys = set(
        zip(
            pool_df["repository"].astype(str),
            pool_df["sha"].astype(str),
        )
    )

    labeled_inside_pool = labeled_keys & pool_keys

    print(
        f"Already reviewed inside pool: "
        f"{len(labeled_inside_pool)}"
    )

    print(
        f"Remaining unlabeled commits: "
        f"{len(pool_df) - len(labeled_inside_pool)}"
    )

    print("\nCommit Labeling Tool Started")
    print(
        "Read reports/labeling_guidelines.md "
        "before assigning labels."
    )

    print(
        "Every decision is saved immediately."
    )

    for index, row in pool_df.iterrows():
        key = get_commit_key(row)

        if key in labeled_keys:
            continue

        reviewed_count = len(
            labeled_keys & pool_keys
        )

        print_commit(
            row=row,
            pool_position=index + 1,
            pool_size=len(pool_df),
            reviewed_count=reviewed_count,
        )

        while True:
            choice = input("\nEnter label: ").strip().lower()

            if choice in VALID_LABELS:
                label = VALID_LABELS[choice]

                labeled_df = save_label(
                    row,
                    label,
                    labeled_df,
                )

                labeled_keys.add(key)

                print(
                    f"\nSaved label: {label}"
                )

                print(
                    f"Progress: "
                    f"{len(labeled_keys & pool_keys)} "
                    f"/ {len(pool_df)} reviewed"
                )

                print_label_distribution(
                    labeled_df[
                        labeled_df.apply(
                            lambda labeled_row: (
                                str(labeled_row["repository"]),
                                str(labeled_row["sha"]),
                            )
                            in pool_keys,
                            axis=1,
                        )
                    ]
                )

                break

            elif choice == "s":
                print(
                    "\nCommit skipped for now. "
                    "It will appear again next time."
                )

                break

            elif choice == "q":
                print_label_distribution(
                    labeled_df
                )

                print(
                    "\nProgress saved. "
                    "Exiting labeling tool."
                )

                return

            else:
                print(
                    "Invalid option. "
                    "Enter u, l, ?, s, or q."
                )

    reviewed_count = len(
        labeled_keys & pool_keys
    )

    print("\n" + "=" * 75)

    print("LABELING POOL COMPLETED")

    print("=" * 75)

    print(
        f"Reviewed commits: "
        f"{reviewed_count} / {len(pool_df)}"
    )

    print_label_distribution(
        labeled_df
    )


if __name__ == "__main__":
    main()