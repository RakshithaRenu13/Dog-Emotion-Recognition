
import numpy as np
import pandas as pd

# Input file
INPUT = (
    "dog_pose_outputs/facial_attention_analysis/"
    "facial_attention_consistency.csv"
)

OUTPUT = (
    "dog_pose_outputs/facial_attention_analysis/"
    "bootstrap_confidence_intervals.csv"
)

N_BOOTSTRAP = 10000
SEED = 42

df = pd.read_csv(INPUT)

# Ensure correct is Boolean
if df["correct"].dtype != bool:
    df["correct"] = (
        df["correct"].astype(str).str.lower() == "true"
    )

metrics = [
    "facial_attention_fraction",
    "facial_attention_enrichment"
]

rng = np.random.default_rng(SEED)


def bootstrap_difference(correct_values, incorrect_values):
    """Difference = correct mean - incorrect mean."""

    correct_values = np.asarray(correct_values, dtype=float)
    incorrect_values = np.asarray(incorrect_values, dtype=float)

    observed = correct_values.mean() - incorrect_values.mean()

    boot_diffs = np.empty(N_BOOTSTRAP)

    for i in range(N_BOOTSTRAP):
        c = rng.choice(
            correct_values,
            size=len(correct_values),
            replace=True
        )
        ic = rng.choice(
            incorrect_values,
            size=len(incorrect_values),
            replace=True
        )

        boot_diffs[i] = c.mean() - ic.mean()

    lower, upper = np.percentile(boot_diffs, [2.5, 97.5])

    return observed, lower, upper


results = []

# Overall and per-emotion analysis
groups = [("overall", df)]

for emotion in sorted(df["true_label"].dropna().unique()):
    groups.append(
        (emotion, df[df["true_label"] == emotion])
    )

for group_name, group in groups:
    for metric in metrics:
        valid = group.dropna(subset=[metric])

        correct = valid.loc[
            valid["correct"], metric
        ].to_numpy()

        incorrect = valid.loc[
            ~valid["correct"], metric
        ].to_numpy()

        if len(correct) < 2 or len(incorrect) < 2:
            print(f"Skipping {group_name}, {metric}: "
                  "insufficient samples")
            continue

        difference, lower, upper = bootstrap_difference(
            correct, incorrect
        )

        results.append({
            "emotion": group_name,
            "metric": metric,
            "correct_n": len(correct),
            "incorrect_n": len(incorrect),
            "correct_mean": correct.mean(),
            "incorrect_mean": incorrect.mean(),
            "difference_correct_minus_incorrect": difference,
            "ci_95_lower": lower,
            "ci_95_upper": upper,
            "includes_zero": lower <= 0 <= upper
        })

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT, index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.5f}".format)

print("\nBOOTSTRAP RESULTS")
print("=" * 100)
print(result_df.to_string(index=False))
print("\nSaved to:", OUTPUT)