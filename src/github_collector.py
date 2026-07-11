import time
from pathlib import Path

import pandas as pd
import requests

from src.config import GITHUB_TOKEN


BASE_URL = "https://api.github.com"

OUTPUT_FILE = Path("data/raw_commits.csv")


HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


def github_get(endpoint, params=None):

    url = f"{BASE_URL}{endpoint}"

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=30,
    )

    if response.status_code == 404:
        raise ValueError(
            f"GitHub resource not found: {endpoint}"
        )

    if response.status_code == 403:
        raise RuntimeError(
            "GitHub API request forbidden. "
            "You may have exceeded the API rate limit."
        )

    response.raise_for_status()

    return response

def check_rate_limit():

    response = github_get("/rate_limit")

    data = response.json()

    core = data["resources"]["core"]

    limit = core["limit"]
    remaining = core["remaining"]
    reset_timestamp = core["reset"]

    reset_time = time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(reset_timestamp),
    )

    print("\nGitHub API Rate Limit")
    print("---------------------")
    print(f"Limit:     {limit}")
    print(f"Remaining: {remaining}")
    print(f"Reset:     {reset_time}")

    return remaining

def get_commits(owner, repo, max_commits=100):

    commits = []

    page = 1

    while len(commits) < max_commits:

        response = github_get(
            f"/repos/{owner}/{repo}/commits",
            params={
                "per_page": min(100, max_commits - len(commits)),
                "page": page,
            },
        )

        batch = response.json()

        if not batch:
            break

        commits.extend(batch)

        page += 1

    return commits[:max_commits]


def get_commit_details(owner, repo, sha):

    response = github_get(
        f"/repos/{owner}/{repo}/commits/{sha}"
    )

    return response.json()


def classify_file(filename):
    filename_lower = filename.lower()

    base_name = filename_lower.rsplit("/", 1)[-1]

    dependency_config_files = {
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        "requirements-docs.txt",
        "constraints.txt",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
        "pipfile",
        "poetry.lock",
    }

    if base_name in dependency_config_files:
        return "config"

    source_extensions = (
        ".py",
        ".java",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".kt",
        ".swift",
    )

    if (
        filename_lower.startswith("test/")
        or filename_lower.startswith("tests/")
        or "/test/" in filename_lower
        or "/tests/" in filename_lower
        or filename_lower.startswith("test_")
        or filename_lower.endswith("_test.py")
        or filename_lower.endswith(".test.js")
    ):
        return "test"

    if filename_lower.endswith(
        (
            ".md",
            ".rst",
            ".txt",
        )
    ):
        return "documentation"

    if filename_lower.endswith(
        (
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
        )
    ):
        return "config"

    if filename_lower.endswith(source_extensions):
        return "source"

    return "other"


def build_commit_row(owner, repo, commit):

    sha = commit["sha"]

    details = get_commit_details(
        owner,
        repo,
        sha,
    )

    commit_data = details.get("commit", {})

    author_data = commit_data.get("author") or {}

    github_author = details.get("author") or {}

    stats = details.get("stats") or {}

    files = details.get("files") or []


    source_files = 0
    test_files = 0
    documentation_files = 0
    config_files = 0


    for file_data in files:

        category = classify_file(
            file_data.get("filename", "")
        )

        if category == "source":
            source_files += 1

        elif category == "test":
            test_files += 1

        elif category == "documentation":
            documentation_files += 1

        elif category == "config":
            config_files += 1


    message = commit_data.get(
        "message",
        "",
    )


    return {
        "repository": f"{owner}/{repo}",

        "sha": sha,

        "author_login": github_author.get("login"),

        "author_name": author_data.get("name"),

        "commit_message": message,

        "commit_date": author_data.get("date"),

        "files_changed": len(files),

        "additions": stats.get("additions", 0),

        "deletions": stats.get("deletions", 0),

        "total_changes": stats.get("total", 0),

        "source_files_changed": source_files,

        "test_files_changed": test_files,

        "doc_files_changed": documentation_files,

        "config_files_changed": config_files,

        "message_length": len(message),

        "message_word_count": len(message.split()),

        "is_merge_commit": len(
            details.get("parents", [])
        ) > 1,
    }

def load_existing_commit_keys():

    if not OUTPUT_FILE.exists():
        return set()

    dataframe = pd.read_csv(
        OUTPUT_FILE,
        usecols=[
            "repository",
            "sha",
        ],
    )

    return set(
        zip(
            dataframe["repository"],
            dataframe["sha"],
        )
    )

def collect_repository( owner, repo, max_commits=100, existing_commit_keys=None,):

    print(
        f"\nCollecting {owner}/{repo}"
    )

    commits = get_commits(
        owner,
        repo,
        max_commits=max_commits,
    )

    rows = []

    if existing_commit_keys is None:
        existing_commit_keys = set()

    for index, commit in enumerate(
        commits,
        start=1,
    ):

        sha = commit["sha"]
        
        repository_name = f"{owner}/{repo}"

        commit_key = (
            repository_name,
            sha,
        )

        if commit_key in existing_commit_keys:

            print(
                f"[{index}/{len(commits)}] "
                f"Skipping existing {sha[:8]}"
            )

            continue

        print(
            f"[{index}/{len(commits)}] "
            f"Fetching {sha[:8]}"
        )

        try:

            row = build_commit_row(
                owner,
                repo,
                commit,
            )

            rows.append(row)

        except requests.RequestException as error:

            print(
                f"Failed commit {sha[:8]}: {error}"
            )


        time.sleep(0.1)


    return rows

def add_temporal_features(rows):

    if not rows:
        return rows

    dataframe = pd.DataFrame(rows)

    dataframe["commit_date"] = pd.to_datetime(
        dataframe["commit_date"],
        utc=True,
        errors="coerce",
    )

    dataframe = dataframe.sort_values(
        ["repository", "author_login", "commit_date"]
    )

    dataframe["hours_since_previous_commit"] = (
        dataframe
        .groupby(["repository", "author_login"])["commit_date"]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )

    dataframe["hours_since_previous_commit"] = (
        dataframe["hours_since_previous_commit"]
        .fillna(-1)
    )

    return dataframe.to_dict("records")

def save_commits(new_rows):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not new_rows:
        print("\nNo new commits to save.")
        return

    new_dataframe = pd.DataFrame(new_rows)

    if OUTPUT_FILE.exists():

        existing_dataframe = pd.read_csv(
            OUTPUT_FILE
        )

        combined_dataframe = pd.concat(
            [
                existing_dataframe,
                new_dataframe,
            ],
            ignore_index=True,
        )

    else:

        combined_dataframe = new_dataframe


    before_count = len(combined_dataframe)

    combined_dataframe = (
        combined_dataframe
        .drop_duplicates(
            subset=[
                "repository",
                "sha",
            ],
            keep="last",
        )
    )

    duplicates_removed = (
        before_count
        - len(combined_dataframe)
    )


    combined_dataframe = (
        combined_dataframe
        .sort_values(
            [
                "repository",
                "commit_date",
            ]
        )
        .reset_index(drop=True)
    )


    combined_dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    print(
        f"\nDataset saved successfully."
    )

    print(
        f"Total unique commits: "
        f"{len(combined_dataframe)}"
    )

    print(
        f"Duplicates removed: "
        f"{duplicates_removed}"
    )


def main():

    remaining = check_rate_limit()

    if remaining < 500:

        print(
            "\nNot enough API requests remaining."
        )

        return


    repositories = [

        ("django", "django"),

        ("pallets", "flask"),

        ("psf", "requests"),

        ("fastapi", "fastapi"),

        ("encode", "httpx"),

        ("tiangolo", "sqlmodel"),

        ("pandas-dev", "pandas"),

        ("scikit-learn", "scikit-learn"),

        ("keras-team", "keras"),

        ("pytest-dev", "pytest"),

    ]


    existing_commit_keys = (
        load_existing_commit_keys()
    )


    print(
        f"\nExisting dataset commits: "
        f"{len(existing_commit_keys)}"
    )


    for owner, repo in repositories:

        rows = collect_repository(

            owner,

            repo,

            max_commits=200,

            existing_commit_keys=existing_commit_keys,

        )


        rows = add_temporal_features(rows)


        save_commits(rows)


        for row in rows:

            existing_commit_keys.add(

                (
                    row["repository"],
                    row["sha"],
                )

            )


        remaining = check_rate_limit()


        if remaining < 500:

            print(
                "\nStopping collection because "
                "the API rate limit is low."
            )

            break


    print(
        "\nCollection process finished."
    )


if __name__ == "__main__":
    main()
