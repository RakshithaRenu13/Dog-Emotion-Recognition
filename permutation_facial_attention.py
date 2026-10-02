
import numpy as np
import pandas as pd

INPUT = (
    "dog_pose_outputs/facial_attention_analysis/"
    "facial_attention_consistency.csv"
)
OUTPUT = (
    "dog_pose_outputs/facial_attention_analysis/"
    "permutation_results.csv"
)

N_PERMUTATIONS = 10000
SEED = 42
rng = np.random.default_rng(SEED)

df = pd.read_csv(INPUT)
df["correct"] = (
    df["correct"].astype(str).str.lower() == "true"
)

metrics = [
    "facial_attention_fraction",
    "facial_attention_enrichment"
]

results = []

for emotion in sorted(df["true_label"].dropna().unique()):
    group = df[df["true_label"] == emotion]

    for metric in metrics:
        valid = group.dropna(subset=[metric])
        values = valid[metric].to_numpy(float)
        labels = valid["correct"].to_numpy(bool)

        n_correct = labels.sum()
        n_incorrect = len(labels) - n_correct

        if n_correct < 2 or n_incorrect < 2:
            continue

        observed = (
            values[labels].mean() -
            values[~labels].mean()
        )

        permuted_diffs = np.empty(N_PERMUTATIONS)

        for i in range(N_PERMUTATIONS):
            shuffled = rng.permutation(labels)
            permuted_diffs[i] = (
                values[shuffled].mean() -
                values[~shuffled].mean()
            )

        p = (
            1 + np.sum(
                np.abs(permuted_diffs) >= abs(observed)
            )
        ) / (N_PERMUTATIONS + 1)

        results.append({
            "emotion": emotion,
            "metric": metric,
            "correct_n": n_correct,
            "incorrect_n": n_incorrect,
            "difference": observed,
            "permutation_p": p
        })

result = pd.DataFrame(results)

# Holm-Bonferroni correction
m = len(result)
order = np.argsort(result["permutation_p"].to_numpy())
p_sorted = result["permutation_p"].to_numpy()[order]

adjusted_sorted = np.maximum.accumulate(
    np.minimum(1.0, (m - np.arange(m)) * p_sorted)
)

adjusted = np.empty(m)
adjusted[order] = adjusted_sorted

result["holm_adjusted_p"] = adjusted
result["significant"] = adjusted < 0.05

result = result.sort_values(
    ["metric", "emotion"]
).reset_index(drop=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.6f}".format)

print("\nPERMUTATION TEST RESULTS")
print("=" * 100)
print(result.to_string(index=False))

result.to_csv(OUTPUT, index=False)
print("\nSaved to:", OUTPUT)