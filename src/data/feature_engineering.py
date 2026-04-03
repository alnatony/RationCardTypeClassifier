import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib, os

df = pd.read_csv("data/cleaned_dataset.csv")
print(f"Loaded: {len(df)} rows")

# ── Better Feature Engineering ────────────────────────────────────────────────

# Income features
df["annual_income"]       = df["monthly_income"] * 12
df["income_per_person"]   = df["annual_income"] / df["family_size"].replace(0, 1)
df["income_log"]          = np.log1p(df["monthly_income"])  # log scale helps a lot

# Family features
df["dependency_ratio"]    = (df["family_size"] - df["earning_members"]) / df["family_size"].replace(0, 1)
df["earning_ratio"]       = df["earning_members"] / df["family_size"].replace(0, 1)

# Asset features
df["asset_score"]         = df["land_area"] + (df["vehicle_owned"] * 10) + (df["tax_paid"] * 20)
df["land_per_person"]     = df["land_area"] / df["family_size"].replace(0, 1)
df["has_land"]            = (df["land_area"] > 0).astype(int)
df["has_high_land"]       = (df["land_area"] > 50).astype(int)

# Income brackets — very important for separating AAY/PHH/NPS
df["is_zero_income"]      = (df["monthly_income"] == 0).astype(int)
df["is_very_low_income"]  = (df["monthly_income"] < 1000).astype(int)
df["is_low_income"]       = (df["monthly_income"] < 3000).astype(int)
df["is_medium_income"]    = ((df["monthly_income"] >= 3000) & (df["monthly_income"] < 10000)).astype(int)
df["is_high_income"]      = (df["monthly_income"] >= 10000).astype(int)

# Single person household flag (NPI indicator)
df["is_single_person"]    = (df["family_size"] == 1).astype(int)

# Combined poverty indicators
df["poverty_score"]       = (
    (df["monthly_income"] < 1000).astype(int) +
    (df["land_area"] < 5).astype(int) +
    (df["vehicle_owned"] == 0).astype(int) +
    (df["tax_paid"] == 0).astype(int)
)

# ── One-Hot Encoding ──────────────────────────────────────────────────────────
df = pd.get_dummies(df, columns=["housing_type", "region_type", "employment_type"])

# ── Label Encoding ────────────────────────────────────────────────────────────
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])
print("Label classes:", le.classes_)
print(df["label_encoded"].value_counts())

df.drop(columns=["label", "monthly_income"], inplace=True)

print(f"Final shape: {df.shape}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
df.to_csv("data/features_dataset.csv", index=False)
joblib.dump(le, "models/label_encoder.pkl")
feature_cols = [c for c in df.columns if c != "label_encoded"]
joblib.dump(feature_cols, "models/feature_columns.pkl")
print(f"✅ Done! Saved {len(feature_cols)} features")



