import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Step 1: Load Dataset ---
print("Fetching Adult Dataset...")
adult = fetch_ucirepo(id=2) 
X = adult.data.features
y = adult.data.targets

# প্রসেসিং: ক্যাটাগরিক্যাল ডেটাকে হ্যান্ডেল করা
X = pd.get_dummies(X).astype(float)
y = LabelEncoder().fit_transform(y.values.ravel())

# ট্রেন-টেস্ট স্প্লিট
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# স্কেলিং (SVM এর জন্য অত্যন্ত জরুরি)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 2: Extract Support Vectors ---
print("Training Initial SVM for SV Extraction...")
# ছোট সাবসেট ব্যবহার করছি দ্রুত এক্সট্রাকশনের জন্য (রিসার্চে এটি Coreset selection হিসেবে পরিচিত)
svm_model = SVC(kernel='rbf', C=1.0)
svm_model.fit(X_train_scaled[:10000], y_train[:10000]) 

sv_indices = svm_model.support_
X_sv = X_train_scaled[sv_indices]
y_sv = y_train[sv_indices]
print(f"Original SVs Extracted: {len(X_sv)}")

# --- Step 3: Adversarial Margin Augmentation (AMA) ---
def apply_ama(X_sv, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X_sv.shape)
    X_sv_augmented = X_sv + noise
    return X_sv_augmented

X_sv_syn = apply_ama(X_sv) # সিন্থেটিক সাপোর্ট ভেক্টর তৈরি
X_combined_sv = np.vstack([X_sv, X_sv_syn])
y_combined_sv = np.concatenate([y_sv, y_sv])
print(f"Total SVs after AMA (Union Set): {len(X_combined_sv)}")

# --- Step 4: Smart Sampling on Remaining Data ---
# আমরা এখানে সিম্পল র‍্যান্ডম স্যাম্পলিং দেখাচ্ছি, রিসার্চে আপনি K-Means ব্যবহার করতে পারেন
remaining_indices = np.setdiff1d(np.arange(len(X_train_scaled)), sv_indices)
X_remaining = X_train_scaled[remaining_indices]
y_remaining = y_train[remaining_indices]

# মাত্র ৫% রিমেইনিং ডেটা স্যাম্পল নিচ্ছি
sample_size = int(len(X_remaining) * 0.05)
idx = np.random.choice(len(X_remaining), sample_size, replace=False)
X_smart_sample = X_remaining[idx]
y_smart_sample = y_remaining[idx]

# --- Step 5: Merge & Final Distillation ---
X_distilled = np.vstack([X_combined_sv, X_smart_sample])
y_distilled = np.concatenate([y_combined_sv, y_smart_sample])
print(f"Condensed Dataset Size: {len(X_distilled)} (Original was {len(X_train)})")

# --- Step 6: Train Multiple Models (Ensemble) ---
print("\nTraining Final Ensemble Models on Distilled Data...")

# Model A: Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_distilled, y_distilled)
y_pred_rf = rf.predict(X_test_scaled)

# Model B: XGBoost
xgb = XGBClassifier()
xgb.fit(X_distilled, y_distilled)
y_pred_xgb = xgb.predict(X_test_scaled)

# --- Results Comparison ---
print("\n--- PERFORMANCE REPORT ---")
print(f"Random Forest Accuracy (Distilled Data): {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"XGBoost Accuracy (Distilled Data): {accuracy_score(y_test, y_pred_xgb):.4f}")
