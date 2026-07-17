import pandas as pd

# Load dataset
df = pd.read_csv("data/labeled_commits_final.csv")

# Replace missing values with empty strings
text_columns = [
    "commit_message",
    "changed_filenames",
    "file_summary",
    "patch_summary"
]

for col in text_columns:
    df[col] = df[col].fillna("")

# Create one combined text column
df["combined_text"] = (
    df["commit_message"] + " " +
    df["changed_filenames"] + " " +
    df["file_summary"] + " " +
    df["patch_summary"]
)

# Show the first combined text
print("=" * 80)
print(df["combined_text"].iloc[0][:1000])
print("commit_message:", (df["commit_message"].str.strip() != "").sum())
print("changed_filenames:", (df["changed_filenames"].str.strip() != "").sum())
print("file_summary:", (df["file_summary"].str.strip() != "").sum())
print("patch_summary:", (df["patch_summary"].str.strip() != "").sum())
print("=" * 80)
print("\nCombined text length:", len(df["combined_text"].iloc[0]))

df.to_csv("data/processed_commits.csv", index=False)

print("\n✅ Processed dataset saved as data/processed_commits.csv")
