import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pytesseract
import cv2
import re
import tempfile
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

model     = joblib.load("models/ration_model.pkl")
le        = joblib.load("models/label_encoder.pkl")
feat_cols = joblib.load("models/feature_columns.pkl")

st.set_page_config(page_title="റേഷൻ കാർഡ് വർഗ്ഗീകരണം", page_icon="🪪", layout="centered")

st.title("🪪 റേഷൻ കാർഡ് വിഭാഗം പ്രവചിക്കുക")
st.caption("കേരള സർക്കാർ — തീരുമാന സഹായ സംവിധാനം")
st.divider()

# ── Step 1: Upload ────────────────────────────────────────────────────────────
st.subheader("ഘട്ടം 1 — വരുമാന സർട്ടിഫിക്കറ്റ് അപ്‌ലോഡ് ചെയ്യുക")
st.caption("(FORM 10C — ഐച്ഛികം)")
uploaded_file = st.file_uploader("സർട്ടിഫിക്കറ്റ് ചിത്രം തിരഞ്ഞെടുക്കുക", type=["jpg","jpeg","png"])

extracted_income = None

if uploaded_file:
    st.image(uploaded_file, caption="അപ്‌ലോഡ് ചെയ്ത സർട്ടിഫിക്കറ്റ്", width=400)
    with st.spinner("സർട്ടിഫിക്കറ്റ് വായിക്കുന്നു..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        img = cv2.imread(tmp_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(gray, lang="eng")
        os.unlink(tmp_path)
        for pattern in [r'[₹%Rs\.]+\s*\.?\s*([0-9,]+)', r'[Ii]ncome[^0-9]*([0-9,]+)', r'[Aa]nnual[^0-9]*([0-9,]+)', r'([0-9]{5,7})']:
            for match in re.findall(pattern, text):
                raw = match.replace(",","").strip()
                try:
                    amount = int(raw)
                    if 10000 <= amount <= 5000000:
                        extracted_income = amount
                        break
                except: continue
            if extracted_income: break
    if extracted_income:
        st.success(f"✅ വാർഷിക വരുമാനം കണ്ടെത്തി: ₹{extracted_income} → പ്രതിമാസം: ₹{extracted_income//12}")
    else:
        st.warning("⚠️ വരുമാനം സ്വയം കണ്ടെത്താനായില്ല. താഴെ നൽകുക.")

st.divider()

# ── Step 2: Form ──────────────────────────────────────────────────────────────
st.subheader("ഘട്ടം 2 — അപേക്ഷകന്റെ വിവരങ്ങൾ നൽകുക")

default_income = (extracted_income // 12) if extracted_income else 0

col1, col2 = st.columns(2)

with col1:
    monthly_income = st.number_input("പ്രതിമാസ വരുമാനം (₹)", 0, 500000, default_income)
    family_size    = st.number_input("കുടുംബാംഗങ്ങളുടെ എണ്ണം", 1, 20, 4)
    earning_members= st.number_input("വരുമാനം ഉള്ളവരുടെ എണ്ണം", 0, 20, 1)
    land_area      = st.number_input("ഭൂമി (സെന്റ്)", 0.0, 500.0, 0.0)

with col2:
    housing_ml = st.selectbox("വീടിന്റെ തരം", [
        "മറ്റുളളവ",
        "ഭാഗികമായി പൂർത്തിയായത് / ജീർണ്ണിച്ചത്",
        "കുടിൽ",
        "ഓല/ പുല്ല് മേഞ്ഞത്",
        "nil"
    ])
    employment_ml = st.selectbox("തൊഴിൽ വിവരം", [
        "ഗൃഹഭരണം",
        "തൊഴിൽരഹിതർ",
        "കൂലി",
        "കൃഷി",
        "സർവീസ് (സർക്കാർ)",
        "പ്രൈവറ്റ്",
        "സർവീസ് പെൻഷണർ",
        "കന്യാസ്ത്രീ / പുരോഹിതൻ / സന്ന്യാസം (NPI)",
        "ബീഡിതൊഴിലാളി",
        "ബാധകമല്ല",
        "അദ്ധ്യാപനം",
        "നഴ്സ്",
        "വിദ്യാർത്ഥി",
        "സ്വയംതൊഴിൽ",
        "പൊതുപ്രവർത്തനം",
    ])
    region_ml  = st.selectbox("പ്രദേശം", ["നഗരം (Urban)", "ഗ്രാമം (Rural)"])
    vehicle_ml = st.selectbox("വാഹനം ഉണ്ടോ?", ["ഇല്ല (No)", "ഉണ്ട് (Yes)"])
    tax_ml     = st.selectbox("ആദായ നികുതി അടക്കുന്നുണ്ടോ?", ["ഇല്ല (No)", "ഉണ്ട് (Yes)"])

st.divider()

# ── Step 3: Predict ───────────────────────────────────────────────────────────
st.subheader("ഘട്ടം 3 — വിഭാഗം പ്രവചിക്കുക")

if st.button("🔍 റേഷൻ കാർഡ് വിഭാഗം കണ്ടെത്തുക", use_container_width=True):

    # Map Malayalam selections to English training values
    housing_map = {
        "മറ്റുളളവ":                                        "other",
        "ഭാഗികമായി പൂർത്തിയായത് / ജീർണ്ണിച്ചത്":          "dilapidated",
        "കുടിൽ":                                           "hut",
        "ഓല/ പുല്ല് മേഞ്ഞത്":                              "thatched",
        "nil":                                             "nil",
    }
    employment_map = {
        "ഗൃഹഭരണം":                                        "homemaker",
        "തൊഴിൽരഹിതർ":                                     "unemployed",
        "കൂലി":                                            "daily_wage",
        "കൃഷി":                                            "farmer",
        "സർവീസ് (സർക്കാർ)":                               "government",
        "പ്രൈവറ്റ്":                                       "private",
        "സർവീസ് പെൻഷണർ":                                  "pensioner",
        "കന്യാസ്ത്രീ / പുരോഹിതൻ / സന്ന്യാസം":       "institutional",
        "ബീഡിതൊഴിലാളി":                                   "bidi_worker",
        "ബാധകമല്ല":                                        "not_applicable",
        "അദ്ധ്യാപനം":                                      "teacher",
        "നഴ്സ്":                                           "nurse",
        "വിദ്യാർത്ഥി":                                     "student",
        "സ്വയംതൊഴിൽ":                                     "self_employed",
        "പൊതുപ്രവർത്തനം":                                  "public_worker",
    }

    housing    = housing_map.get(housing_ml, "other")
    employment = employment_map.get(employment_ml, "other")
    region     = "urban" if "Urban" in region_ml else "rural"
    vehicle    = 1 if "Yes" in vehicle_ml else 0
    tax        = 1 if "Yes" in tax_ml else 0

    # Feature engineering
    annual_income     = monthly_income * 12
    income_per_person = annual_income / max(family_size, 1)
    dependency_ratio  = (family_size - earning_members) / max(family_size, 1)
    earning_ratio     = earning_members / max(family_size, 1)
    asset_score       = land_area + (vehicle * 10) + (tax * 20)
    land_per_person   = land_area / max(family_size, 1)
    has_land          = 1 if land_area > 0 else 0
    has_high_land     = 1 if land_area > 50 else 0
    is_zero_income    = 1 if monthly_income == 0 else 0
    is_very_low       = 1 if monthly_income < 1000 else 0
    is_low            = 1 if monthly_income < 3000 else 0
    is_medium         = 1 if 3000 <= monthly_income < 10000 else 0
    is_high           = 1 if monthly_income >= 10000 else 0
    is_single         = 1 if family_size == 1 else 0
    poverty_score     = int(monthly_income < 1000) + int(land_area < 5) + int(vehicle == 0) + int(tax == 0)

    # Build row — start with zeros for all feature columns
    row = {col: 0 for col in feat_cols}

    # Fill numeric features
    row["annual_income"]       = annual_income
    row["income_per_person"]   = income_per_person
    row["income_log"]          = np.log1p(monthly_income)
    row["dependency_ratio"]    = dependency_ratio
    row["earning_ratio"]       = earning_ratio
    row["asset_score"]         = asset_score
    row["land_per_person"]     = land_per_person
    row["has_land"]            = has_land
    row["has_high_land"]       = has_high_land
    row["is_zero_income"]      = is_zero_income
    row["is_very_low_income"]  = is_very_low
    row["is_low_income"]       = is_low
    row["is_medium_income"]    = is_medium
    row["is_high_income"]      = is_high
    row["is_single_person"]    = is_single
    row["poverty_score"]       = poverty_score
    row["land_area"]           = land_area
    row["family_size"]         = family_size
    row["earning_members"]     = earning_members
    row["vehicle_owned"]       = vehicle
    row["tax_paid"]            = tax

    # One-hot encode — set the matching column to 1
    housing_col    = f"housing_type_{housing}"
    region_col     = f"region_type_{region}"
    employment_col = f"employment_type_{employment}"

    if housing_col in row:    row[housing_col]    = 1
    if region_col in row:     row[region_col]     = 1
    if employment_col in row: row[employment_col] = 1

    # Build dataframe aligned to training columns
    df_input = pd.DataFrame([row])[feat_cols]

    # Predict
    pred  = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]
    label = le.inverse_transform([pred])[0]
    conf  = proba[pred] * 100

    # Malayalam label explanations
    label_ml = {
        "AAY":  "AAY — അന്ത്യോദയ അന്ന യോജന (ഏറ്റവും ദരിദ്രർ)",
        "PHH":  "PHH — മുൻഗണനാ കുടുംബം (ദാരിദ്ര്യ രേഖയ്ക്ക് താഴെ)",
        "NPS":  "NPS — മുൻഗണനേതര പദ്ധതി (ഭാഗിക ആനുകൂല്യം)",
        "NPI":  "NPI — സ്ഥാപന വിഭാഗം (കന്യാസ്ത്രീ / ആശ്രമം)",
        "NPNS": "NPNS — മുൻഗണനേതര (ദാരിദ്ര്യ രേഖയ്ക്ക് മുകളിൽ)",
    }
    color = {"AAY":"🟡","PHH":"🔴","NPS":"🔵","NPI":"🟤","NPNS":"⚪"}

    st.success(f"### {color.get(label,'⚪')} പ്രവചിത വിഭാഗം: **{label}**")
    st.info(f"ℹ️ {label_ml.get(label,'')}")
    st.metric("വിശ്വാസ്യത (Confidence)", f"{conf:.1f}%")

    # Probability chart with Malayalam labels
    prob_df = pd.Series(proba * 100, index=le.classes_).rename("സാധ്യത (%)")
    st.bar_chart(prob_df)

    st.warning("⚠️ ഇത് ഒരു തീരുമാന സഹായ ഉപകരണം മാത്രമാണ്. ഔദ്യോഗിക സ്ഥിരീകരണം ആവശ്യമാണ്.")