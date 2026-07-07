from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/cleaned_commits.csv")

REPORT_FILE = Path("reports/feature_audit.txt")


IDENTIFIER_COLUMNS = [

    "repository",

    "sha",

    "author_login",

    "author_name",

]


TEXT_COLUMNS = [

    "commit_message",

]


BOOLEAN_COLUMNS = [

    "is_merge_commit",

]


NUMERIC_COLUMNS = [

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


def load_dataset():

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} commits.")

    return df


def audit_expected_columns(df):

    expected_columns = (

        IDENTIFIER_COLUMNS

        + TEXT_COLUMNS

        + BOOLEAN_COLUMNS

        + NUMERIC_COLUMNS

    )

    missing_columns = [

        column

        for column in expected_columns

        if column not in df.columns

    ]

    unexpected_columns = [

        column

        for column in df.columns

        if column not in expected_columns

    ]

    return missing_columns, unexpected_columns


def calculate_missing_values(df):

    return (

        df.isna()

        .sum()

        .sort_values(ascending=False)

    )


def calculate_unique_values(df):

    return (

        df.nunique(dropna=False)

        .sort_values()

    )


def calculate_skewness(df):

    available_columns = [

        column

        for column in NUMERIC_COLUMNS

        if column in df.columns

    ]

    return (

        df[available_columns]

        .skew()

        .sort_values(ascending=False)

    )


def find_high_correlations(df, threshold=0.80):

    available_columns = [

        column

        for column in NUMERIC_COLUMNS

        if column in df.columns

    ]

    correlation = (

        df[available_columns]

        .corr()

        .abs()

    )

    pairs = []

    for index, first_column in enumerate(available_columns):

        for second_column in available_columns[index + 1:]:

            value = correlation.loc[

                first_column,

                second_column,

            ]

            if value >= threshold:

                pairs.append(

                    (

                        first_column,

                        second_column,

                        value,

                    )

                )

    return sorted(

        pairs,

        key=lambda item: item[2],

        reverse=True,

    )


def calculate_outlier_counts(df):

    results = []

    for column in NUMERIC_COLUMNS:

        if column not in df.columns:

            continue

        series = df[column].dropna()

        q1 = series.quantile(0.25)

        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr

        upper_bound = q3 + 1.5 * iqr

        outlier_count = (

            (series < lower_bound)

            | (series > upper_bound)

        ).sum()

        results.append(

            (

                column,

                int(outlier_count),

                lower_bound,

                upper_bound,

            )

        )

    return results


def generate_recommendations(

    missing_columns,

    high_correlations,

    skewness,

):

    recommendations = []


    if missing_columns:

        recommendations.append(

            "Fix missing expected columns before modeling."

        )


    recommendations.append(

        "Exclude repository, SHA, author_login, and author_name "

        "from direct model features."

    )


    recommendations.append(

        "Keep commit_message for NLP/text-derived features."

    )


    recommendations.append(

        "Do not automatically delete statistical outliers; "

        "use robust preprocessing and log transformations."

    )


    highly_skewed = skewness[skewness.abs() >= 2].index.tolist()


    if highly_skewed:

        recommendations.append(

            "Consider log1p transformation for highly skewed "

            "non-negative features: "

            + ", ".join(highly_skewed)

        )


    if high_correlations:

        recommendations.append(

            "Review redundant highly correlated features before "

            "final model training."

        )


    recommendations.append(

        "Evaluate with a repository-grouped split in addition "

        "to a random stratified split."

    )


    return recommendations


def save_report(

    df,

    missing_columns,

    unexpected_columns,

    missing_values,

    unique_values,

    skewness,

    high_correlations,

    outliers,

    recommendations,

):

    REPORT_FILE.parent.mkdir(

        parents=True,

        exist_ok=True,

    )


    with open(REPORT_FILE, "w") as file:

        file.write("FINAL FEATURE AUDIT\n")

        file.write("=" * 70 + "\n\n")


        file.write(f"Rows: {len(df)}\n")

        file.write(f"Columns: {len(df.columns)}\n\n")


        file.write("MISSING EXPECTED COLUMNS\n")

        file.write("-" * 70 + "\n")

        file.write(str(missing_columns) + "\n\n")


        file.write("UNEXPECTED COLUMNS\n")

        file.write("-" * 70 + "\n")

        file.write(str(unexpected_columns) + "\n\n")


        file.write("MISSING VALUES\n")

        file.write("-" * 70 + "\n")

        file.write(missing_values.to_string())

        file.write("\n\n")


        file.write("UNIQUE VALUES\n")

        file.write("-" * 70 + "\n")

        file.write(unique_values.to_string())

        file.write("\n\n")


        file.write("SKEWNESS\n")

        file.write("-" * 70 + "\n")

        file.write(skewness.to_string())

        file.write("\n\n")


        file.write("HIGH CORRELATIONS >= 0.80\n")

        file.write("-" * 70 + "\n")


        if high_correlations:

            for first, second, value in high_correlations:

                file.write(

                    f"{first} <-> {second}: "

                    f"{value:.4f}\n"

                )

        else:

            file.write("None\n")


        file.write("\nOUTLIER COUNTS USING IQR\n")

        file.write("-" * 70 + "\n")


        for column, count, lower, upper in outliers:

            file.write(

                f"{column}: "

                f"{count} outliers "

                f"(bounds {lower:.4f}, {upper:.4f})\n"

            )


        file.write("\nRECOMMENDATIONS\n")

        file.write("-" * 70 + "\n")


        for index, recommendation in enumerate(

            recommendations,

            start=1,

        ):

            file.write(

                f"{index}. {recommendation}\n"

            )


def main():

    df = load_dataset()


    missing_columns, unexpected_columns = (

        audit_expected_columns(df)

    )


    missing_values = calculate_missing_values(df)

    unique_values = calculate_unique_values(df)

    skewness = calculate_skewness(df)

    high_correlations = find_high_correlations(df)

    outliers = calculate_outlier_counts(df)


    recommendations = generate_recommendations(

        missing_columns,

        high_correlations,

        skewness,

    )


    save_report(

        df=df,

        missing_columns=missing_columns,

        unexpected_columns=unexpected_columns,

        missing_values=missing_values,

        unique_values=unique_values,

        skewness=skewness,

        high_correlations=high_correlations,

        outliers=outliers,

        recommendations=recommendations,

    )


    print("\nFeature audit completed.")

    print(f"Missing expected columns: {missing_columns}")

    print(f"Unexpected columns: {unexpected_columns}")

    print(

        f"High-correlation pairs: "

        f"{len(high_correlations)}"

    )

    print(f"Report saved to {REPORT_FILE}")


if __name__ == "__main__":

    main()