import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "dinov2_svm_tuned.joblib"
)

print("=" * 70)
print("INSPECTING DINOV2 + SVM MODEL")
print("=" * 70)

print("\nModel path:")
print(MODEL_PATH)

model = joblib.load(MODEL_PATH)

print("\nObject type:")
print(type(model))

print("\nModel details:")
print(model)

print("\nAttributes:")

for attr in [
    "n_features_in_",
    "classes_",
    "feature_names_in_",
    "steps",
    "named_steps",
    "coef_",
    "support_"
]:

    if hasattr(model, attr):

        try:
            value = getattr(model, attr)

            if attr == "coef_":
                print(
                    f"{attr}: shape = {value.shape}"
                )

            elif attr == "support_":
                print(
                    f"{attr}: shape = {value.shape}"
                )

            else:
                print(
                    f"{attr}: {value}"
                )

        except Exception as e:

            print(
                f"{attr}: <could not read: {e}>"
            )

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)