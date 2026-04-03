import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/features_dataset.csv")
le = joblib.load("models/label_encoder.pkl")

X = df.drop(columns=["label_encoded"])
y = df["label_encoded"]

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── Train XGBoost (Primary) ───────────────────────────────────────────────────
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ XGBoost Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Confusion Matrix — XGBoost")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
print("✅ Saved confusion_matrix.png")

# ── Feature Importance ────────────────────────────────────────────────────────
feat_imp = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feat_imp.head(15).plot(kind="bar")
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
print("✅ Saved feature_importance.png")

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(xgb, "models/ration_model.pkl")
print("✅ Saved ration_model.pkl")

# ── Compare with Random Forest ────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"\nRandom Forest Accuracy: {rf_acc:.4f}")
print(f"XGBoost Accuracy:       {acc:.4f}")