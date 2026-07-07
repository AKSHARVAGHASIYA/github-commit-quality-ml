# GitHub Commit Quality Labeling Guidelines

## Objective

The objective is to classify GitHub commits based on the evidence available from:

- commit message
- changed files
- additions and deletions
- source-code changes
- test changes
- documentation changes
- configuration changes
- merge status
- commit timing metadata

The labels do not determine whether the developer is good or bad.

The labels describe whether a commit provides sufficient evidence of meaningful project contribution.

---

# LABEL 1: USEFUL

Assign USEFUL when the commit provides clear evidence of meaningful project work.

Examples:

- implementing a feature
- fixing a bug
- adding or improving tests
- refactoring code
- improving documentation meaningfully
- changing configuration for a clear technical reason
- improving performance
- improving security
- updating dependencies with a clear purpose
- meaningful maintenance work

Example commit messages:

- Fix database connection pool leak
- Add validation for student registration
- Implement GitHub commit synchronization
- Refactor authentication middleware
- Add tests for payment processing
- Update API documentation for OAuth flow

A commit should NOT be labeled USEFUL only because it is large.

---

# LABEL 2: LOW_VALUE

Assign LOW_VALUE when the commit provides strong evidence of trivial, noisy, spam-like, or non-informative work.

Examples:

- meaningless commit message
- obvious testing commit
- random text
- typo-only message with negligible changes
- repeated micro-commits with no meaningful evidence
- empty commit
- generated noise with no clear project contribution

Example commit messages:

- wip
- test
- asdf
- aaa
- update
- fix
- changes
- final
- final2

A short commit message alone is NOT sufficient to label a commit LOW_VALUE.

The actual changed files and code statistics must also be considered.

---

# LABEL 3: UNCERTAIN

Assign UNCERTAIN when available evidence is insufficient or conflicting.

Examples:

- vague message but substantial source-code changes
- meaningful message but zero code changes
- very large generated-file commit
- dependency lock-file update without enough context
- merge commits where individual contribution is unclear
- commits that cannot be confidently classified

When unsure, prefer UNCERTAIN instead of guessing.

---

# Important Rules

1. Do not label based only on commit-message length.

2. Do not label based only on number of changed lines.

3. Large commits are not automatically useful.

4. Small commits are not automatically low-value.

5. Evaluate commit message and repository changes together.

6. Do not use author identity to determine the label.

7. Do not use repository identity to determine the label.

8. Merge commits should normally be UNCERTAIN unless there is strong evidence for another label.

9. Generated files and dependency files require careful review.

10. If confidence is low, use UNCERTAIN.

---

# Recommended Initial Dataset

Manually label approximately:

- 150 USEFUL commits
- 150 LOW_VALUE commits
- 100 UNCERTAIN commits

Target:

400 manually reviewed commits.

The first supervised binary model will use:

USEFUL vs LOW_VALUE

UNCERTAIN commits will initially be excluded from training.