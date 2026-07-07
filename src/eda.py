from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = Path("data/cleaned_commits.csv")
FIGURE_DIR = Path("reports/figures")
REPORT_FILE = Path("reports/eda_summary.txt")


def load_dataset():
    df = pd.read_csv(INPUT_FILE)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} cleaned commits.")

    return df


def save_histogram(df, column, filename, upper_quantile=None):
    data = df[column].dropna()

    if upper_quantile is not None:
        upper_limit = data.quantile(upper_quantile)
        data = data[data <= upper_limit]

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=40)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Number of Commits")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=150)
    plt.close()


def plot_repository_median_changes(df):
    values = (
        df.groupby("repository")["total_changes"]
        .median()
        .sort_values()
    )

    plt.figure(figsize=(11, 7))
    values.plot(kind="barh")
    plt.title("Median Total Changes by Repository")
    plt.xlabel("Median Total Changes")
    plt.ylabel("Repository")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "repository_median_changes.png",
        dpi=150,
    )
    plt.close()


def plot_merge_comparison(df):
    values = (
        df.groupby("is_merge_commit")["total_changes"]
        .median()
    )

    plt.figure(figsize=(8, 6))
    values.plot(kind="bar")
    plt.title("Median Total Changes: Merge vs Non-Merge")
    plt.xlabel("Is Merge Commit")
    plt.ylabel("Median Total Changes")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "merge_change_comparison.png",
        dpi=150,
    )
    plt.close()


def plot_correlation_matrix(df):
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

    correlation = df[available_columns].corr()

    plt.figure(figsize=(12, 10))

    image = plt.imshow(
        correlation,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(image)

    plt.xticks(
        range(len(available_columns)),
        available_columns,
        rotation=90,
    )

    plt.yticks(
        range(len(available_columns)),
        available_columns,
    )

    plt.title("Feature Correlation Matrix")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "correlation_matrix.png",
        dpi=150,
    )

    plt.close()

    return correlation


def generate_summary(df, correlation):
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
        "hours_since_previous_commit",
    ]

    available_columns = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    with open(REPORT_FILE, "w") as file:
        file.write("EXPLORATORY DATA ANALYSIS SUMMARY\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Rows: {len(df)}\n")
        file.write(
            f"Repositories: {df['repository'].nunique()}\n"
        )

        file.write(
            f"Merge commits: {df['is_merge_commit'].sum()}\n"
        )

        file.write(
            "Missing timing values: "
            f"{df['hours_since_previous_commit'].isna().sum()}\n\n"
        )

        file.write("SKEWNESS\n")
        file.write("-" * 60 + "\n")

        file.write(
            df[available_columns]
            .skew()
            .sort_values(ascending=False)
            .to_string()
        )

        file.write("\n\nMEDIANS BY REPOSITORY\n")
        file.write("-" * 60 + "\n")

        file.write(
            df.groupby("repository")[
                [
                    "files_changed",
                    "total_changes",
                    "message_length",
                ]
            ]
            .median()
            .to_string()
        )

        file.write("\n\nCORRELATION MATRIX\n")
        file.write("-" * 60 + "\n")

        file.write(correlation.to_string())


def main():
    df = load_dataset()

    save_histogram(
        df,
        "total_changes",
        "total_changes_distribution.png",
        upper_quantile=0.99,
    )

    save_histogram(
        df,
        "files_changed",
        "files_changed_distribution.png",
        upper_quantile=0.99,
    )

    save_histogram(
        df,
        "message_length",
        "message_length_distribution.png",
        upper_quantile=0.99,
    )

    save_histogram(
        df,
        "hours_since_previous_commit",
        "commit_timing_distribution.png",
        upper_quantile=0.99,
    )

    plot_repository_median_changes(df)

    plot_merge_comparison(df)

    correlation = plot_correlation_matrix(df)

    generate_summary(df, correlation)

    print("\nEDA completed successfully.")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Summary saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()