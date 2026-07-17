import pandas as pd

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/labeled_commits_final.csv")

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

# -----------------------------
# Basic Info
# -----------------------------
print(f"Total commits: {len(df)}")

# -----------------------------
# Duplicate SHA
# -----------------------------
duplicate_sha = df["sha"].duplicated().sum()
print(f"Duplicate SHA: {duplicate_sha}")

# -----------------------------
# Missing Values
# -----------------------------
print("\nMissing Values")

columns_to_check = [
    "sha",
    "repository",
    "commit_message",
    "quality_label"
]

for col in columns_to_check:
    if col in df.columns:
        print(f"{col:20}: {df[col].isna().sum()}")

# -----------------------------
# Empty Strings
# -----------------------------
print("\nEmpty Strings")

for col in columns_to_check:
    if col in df.columns:
        empty = (df[col].astype(str).str.strip() == "").sum()
        print(f"{col:20}: {empty}")

# -----------------------------
# Label Distribution
# -----------------------------
print("\nLabel Distribution")

print(df["quality_label"].value_counts())

# -----------------------------
# Unique Labels
# -----------------------------
print("\nUnique Labels")

print(sorted(df["quality_label"].unique()))

# -----------------------------
# Duplicate Rows
# -----------------------------
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# -----------------------------
# Message Length Statistics
# -----------------------------
if "commit_message" in df.columns:
    lengths = df["commit_message"].fillna("").str.len()

    print("\nCommit Message Length")

    print(lengths.describe())

print("\nVerification Finished Successfully.")