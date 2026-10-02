
import numpy as np
import pandas as pd

INPUT = (
    "dog_pose_outputs/facial_attention_analysis/"
    "facial_attention_consistency.csv"
)

OUTPUT = (
    "dog_pose_outputs/facial_attention_analysis/"
    "holm_corrected_results.csv"
)

N_BOOTSTRAP = 10000
SEED = 42

rng = np.random.default_rng(SEED)
df = pd.read_csv(INPUT)

if df["correct"].dtype != bool:
    df["correct"] = (
        df["correct"].astype(str).str.lower() == "true"
    )

metrics = [
    "facial_attention_fraction",
    "facial_attention_enrichment"
]

results = []

# Only the 10 emotion-wise comparisons are included.
for emotion in sorted(df["true_label"].dropna().unique()):
    group = df[df["true_label"] == emotion]

    for metric in metrics:
        valid = group.dropna(subset=[metric])

        correct = valid.loc[valid["correct"], metric].to_numpy(float)
        incorrect = valid.loc[~valid["correct"], metric].to_numpy(float)

        if len(correct) < 2 or len(incorrect) < 2:
            continue

        observed = correct.mean() - incorrect.mean()
        boot = np.empty(N_BOOTSTRAP)

        for i in range(N_BOOTSTRAP):
            c = rng.choice(correct, size=len(correct), replace=True)
            ic = rng.choice(incorrect, size=len(incorrect), replace=True)
            boot[i] = c.mean() - ic.mean()

        # Two-sided bootstrap test using a centered null distribution
        centered = boot - observed
        p_value = (
            1 + np.count_nonzero(np.abs(centered) >= abs(observed))
        ) / (N_BOOTSTRAP + 1)

        lower, upper = np.percentile(boot, [2.5, 97.5])

        results.append({
            "emotion": emotion,
            "metric": metric,
            "correct_n": len(correct),
            "incorrect_n": len(incorrect),
            "difference": observed,
            "ci_95_lower": lower,
            "ci_95_upper": upper,
            "p_value": p_value
        })

result = pd.DataFrame(results)

# Holm-Bonferroni correction
m = len(result)
order = np.argsort(result["p_value"].to_numpy())
p_sorted = result["p_value"].to_numpy()[order]

adjusted_sorted = np.maximum.accumulate(
    np.minimum(1.0, (m - np.arange(m)) * p_sorted)
)

adjusted = np.empty(m)
adjusted[order] = adjusted_sorted

result["holm_adjusted_p"] = adjusted
result["significant_holm_0_05"] = adjusted < 0.05

result = result.sort_values(
    ["metric", "emotion"]
).reset_index(drop=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", "{:.6f}".format)

print("\nHOLM-CORRECTED RESULTS")
print("=" * 120)
print(result.to_string(index=False))

result.to_csv(OUTPUT, index=False)
print("\nSaved to:", OUTPUT)