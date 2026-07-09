import time
from pathlib import Path

import pandas as pd
import requests

from src.config import GITHUB_TOKEN


INPUT_FILE = Path("data/labeling_pool.csv")
OUTPUT_FILE = Path("data/enriched_labeling_pool.csv")

API_BASE_URL = "https://api.github.com"

MAX_PATCH_CHARS_PER_FILE = 1500
MAX_FILES_IN_SUMMARY = 30
REQUEST_DELAY_SECONDS = 0.15

HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


def load_pool():
    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded labeling pool: {len(df)} commits.")

    return df


def load_existing_enriched_data():
    if not OUTPUT_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(OUTPUT_FILE)

    print(
        f"Loaded existing enriched commits: {len(df)}"
    )

    return df


def get_existing_keys(enriched_df):
    if enriched_df.empty:
        return set()

    return set(
        zip(
            enriched_df["repository"].astype(str),
            enriched_df["sha"].astype(str),
        )
    )


def fetch_commit_details(repository, sha):
    url = (
        f"{API_BASE_URL}/repos/"
        f"{repository}/commits/{sha}"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    if response.status_code == 403:
        remaining = response.headers.get(
            "X-RateLimit-Remaining",
            "unknown",
        )

        reset = response.headers.get(
            "X-RateLimit-Reset",
            "unknown",
        )

        raise RuntimeError(
            "GitHub API returned 403. "
            f"Remaining={remaining}, Reset={reset}"
        )

    response.raise_for_status()

    return response.json()


def is_lockfile(filename):
    filename_lower = filename.lower()

    lockfile_names = {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "pipfile.lock",
        "cargo.lock",
        "composer.lock",
    }

    return (
        Path(filename_lower).name
        in lockfile_names
    )


def looks_generated(filename):
    filename_lower = filename.lower()

    generated_patterns = (
        "/dist/",
        "/build/",
        "/generated/",
        ".min.js",
        ".min.css",
    )

    return any(
        pattern in filename_lower
        for pattern in generated_patterns
    )


def build_file_summary(files):
    summaries = []

    for file_data in files[:MAX_FILES_IN_SUMMARY]:
        filename = file_data.get("filename", "")

        summaries.append(
            " | ".join(
                [
                    filename,
                    f"status={file_data.get('status', '')}",
                    f"+{file_data.get('additions', 0)}",
                    f"-{file_data.get('deletions', 0)}",
                ]
            )
        )

    return "\n".join(summaries)


def build_patch_summary(files):
    patch_parts = []

    for file_data in files[:MAX_FILES_IN_SUMMARY]:
        patch = file_data.get("patch")

        if not patch:
            continue

        filename = file_data.get(
            "filename",
            "unknown",
        )

        bounded_patch = patch[
            :MAX_PATCH_CHARS_PER_FILE
        ]

        patch_parts.append(
            f"FILE: {filename}\n"
            f"{bounded_patch}"
        )

    return "\n\n".join(patch_parts)


def extract_enrichment(commit_data):
    files = commit_data.get("files", [])

    filenames = [
        file_data.get("filename", "")
        for file_data in files
    ]

    statuses = [
        file_data.get("status", "")
        for file_data in files
    ]

    lockfile_count = sum(
        is_lockfile(filename)
        for filename in filenames
    )

    generated_file_count = sum(
        looks_generated(filename)
        for filename in filenames
    )

    patch_available_count = sum(
        bool(file_data.get("patch"))
        for file_data in files
    )

    return {
        "changed_filenames": "\n".join(filenames),
        "file_statuses": "\n".join(statuses),
        "file_summary": build_file_summary(files),
        "patch_summary": build_patch_summary(files),
        "api_files_count": len(files),
        "patch_available_count": patch_available_count,
        "lockfile_count": lockfile_count,
        "generated_file_count": generated_file_count,
    }


def save_progress(enriched_df):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


def main():
    pool_df = load_pool()

    enriched_df = load_existing_enriched_data()

    existing_keys = get_existing_keys(
        enriched_df
    )

    total = len(pool_df)

    for index, row in pool_df.iterrows():
        repository = str(row["repository"])
        sha = str(row["sha"])

        key = (repository, sha)

        if key in existing_keys:
            continue

        print(
            f"[{index + 1}/{total}] "
            f"Fetching {repository} {sha[:8]}"
        )

        try:
            commit_data = fetch_commit_details(
                repository,
                sha,
            )

            enrichment = extract_enrichment(
                commit_data
            )

            record = row.to_dict()
            record.update(enrichment)

            enriched_df = pd.concat(
                [
                    enriched_df,
                    pd.DataFrame([record]),
                ],
                ignore_index=True,
            )

            existing_keys.add(key)

            save_progress(enriched_df)

        except requests.RequestException as error:
            print(
                f"Request failed for "
                f"{repository} {sha}: {error}"
            )

        time.sleep(REQUEST_DELAY_SECONDS)

    print("\nEnrichment completed.")

    print(
        f"Enriched commits saved: "
        f"{len(enriched_df)}"
    )

    print(
        f"Output file: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()