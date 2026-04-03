import pandas as pd

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_excel("data/merged_output.xlsx")
print(f"Step 1 - Loaded: {len(df)} rows")

# ── 2. Drop useless columns ───────────────────────────────────────────────────
df.drop(columns=["Sl No.", "vehicle_owned (Yes/No)", "Income tax payer  (Yes/No)"], inplace=True)

# ── 3. Rename columns ─────────────────────────────────────────────────────────
df.columns = ["monthly_income", "family_size", "earning_members", "housing_type",
              "land_area", "employment_type", "vehicle_owned", "tax_paid",
              "region_type", "label"]
print(f"Step 2 - After rename: {len(df)} rows")

# ── 4. Map housing_type using the EXACT values from the file ──────────────────
housing_map = {
    "മറ്റുളളവ": "other",
    "nil": "nil",
    "ഭാഗികമായി പൂര്‍ത്തിയായത് / ജീര്‍ണ്ണിച്ചത്": "dilapidated",
    "കുടില്‍": "hut",
    "ഓല/ പുല്ല് മേഞ്ഞത്": "thatched",
}
df["housing_type"] = df["housing_type"].map(housing_map).fillna("other")
print(f"Step 3 - After housing map: {len(df)} rows")

# ── 5. Map employment_type ────────────────────────────────────────────────────
def map_employment(val):
    val = str(val)
    if "കന്യാസ്" in val or "പുരോഹിതന്‍" in val or "പൂജാരി" in val or "വേദാ" in val:
        return "institutional"
    elif "സര്‍വീസ്" in val or "സർവീസ്" in val:
        return "government"
    elif "കൂലി" in val:
        return "daily_wage"
    elif "തൊഴില്‍" in val or "തൊഴിൽ" in val:
        return "unemployed"
    elif "ഗൃഹഭരണം" in val or "വീട്ടുജോലി" in val:
        return "homemaker"
    elif "കൃഷി" in val:
        return "farmer"
    elif "പ്രൈവ" in val:
        return "private"
    elif "ബാധകമല്ല" in val:
        return "not_applicable"
    elif "പെന്‍ഷ" in val or "പെൻഷ" in val:
        return "pensioner"
    elif "ബീഡി" in val:
        return "bidi_worker"
    elif "അദ്ധ്യാപ" in val:
        return "teacher"
    elif "നഴ്സ്" in val:
        return "nurse"
    elif "വിദ്യാര്‍" in val:
        return "student"
    elif "സ്വയംതൊഴില്‍" in val:
        return "self_employed"
    elif "പൊതുപ്രവര്‍" in val:
        return "public_worker"
    elif "nil" in val.lower():
        return "nil"
    else:
        return "other"

df["employment_type"] = df["employment_type"].apply(map_employment)
print(f"Step 4 - After employment map: {len(df)} rows")
print(df["employment_type"].value_counts())

# ── 6. Fix land_area ──────────────────────────────────────────────────────────
df["land_area"] = df["land_area"].replace("nil", 0)
df["land_area"] = pd.to_numeric(df["land_area"], errors="coerce").fillna(0)

# ── 7. Binary encode Y/N ──────────────────────────────────────────────────────
df["vehicle_owned"] = df["vehicle_owned"].map({"Y": 1, "N": 0}).fillna(0).astype(int)
df["tax_paid"]      = df["tax_paid"].map({"Y": 1, "N": 0}).fillna(0).astype(int)

# ── 8. Normalize region_type ──────────────────────────────────────────────────
df["region_type"] = df["region_type"].str.strip().str.lower()

# ── 9. Save ───────────────────────────────────────────────────────────────────

df.to_csv("data/cleaned_dataset.csv", index=False)  


print(f"\n✅ Saved! Final rows: {len(df)} (should be 25000)")
