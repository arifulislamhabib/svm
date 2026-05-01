import numpy as np
import pandas as pd
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. Load Dataset
# -----------------------------
# Expect CSV with columns: text,label
df = pd.read_csv("data.csv")

texts = df["text"].astype(str).values
labels = df["label"].values

# -----------------------------
# 2. Feature Extraction
# -----------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(texts)

# -----------------------------
# 3. Train SVM
# -----------------------------
svm = SVC(kernel="linear", probability=True)
svm.fit(X, labels)

# -----------------------------
# 4. Extract Support Vectors
# -----------------------------
sv_indices = svm.support_
SV_texts = texts[sv_indices]
SV_labels = labels[sv_indices]

# -----------------------------
# 5. SG-SV (Stability-based SV)
# -----------------------------
def get_stable_sv(X, y, runs=5):
    freq = np.zeros(len(X))
    for _ in range(runs):
        idx = np.random.choice(len(X), len(X), replace=True)
        svm = SVC(kernel="linear")
        svm.fit(X[idx], y[idx])
        freq[svm.support_] += 1
    stable = np.where(freq >= (runs // 2))[0]
    return stable

stable_indices = get_stable_sv(X, labels)
stable_texts = texts[stable_indices]
stable_labels = labels[stable_indices]

# -----------------------------
# 6. NASP (Remove Noisy SV)
# -----------------------------
# remove SV with low confidence
probs = svm.predict_proba(X[sv_indices])
confidence = np.max(probs, axis=1)

clean_mask = confidence > 0.55
clean_SV_texts = SV_texts[clean_mask]
clean_SV_labels = SV_labels[clean_mask]

# -----------------------------
# 7. AMA (Adversarial Augmentation)
# -----------------------------
def augment_text(text):
    words = text.split()
    if len(words) > 1:
        i = random.randint(0, len(words)-1)
        words[i] = words[i] + "!"
    return " ".join(words)

aug_texts = [augment_text(t) for t in clean_SV_texts]
aug_labels = clean_SV_labels.copy()

# -----------------------------
# 8. GeoSem Smart Sampling
# -----------------------------
# sample non-SV data with diversity
non_sv_mask = np.ones(len(texts), dtype=bool)
non_sv_mask[sv_indices] = False

non_sv_texts = texts[non_sv_mask]
non_sv_labels = labels[non_sv_mask]

sample_size = int(len(non_sv_texts) * 0.3)
sample_idx = np.random.choice(len(non_sv_texts), sample_size, replace=False)

sampled_texts = non_sv_texts[sample_idx]
sampled_labels = non_sv_labels[sample_idx]

# -----------------------------
# 9. Merge Dataset
# -----------------------------
final_texts = list(clean_SV_texts) + list(aug_texts) + list(sampled_texts)
final_labels = list(clean_SV_labels) + list(aug_labels) + list(sampled_labels)

# -----------------------------
# 10. Train Multiple Models
# -----------------------------
X_final = vectorizer.fit_transform(final_texts)

X_train, X_test, y_train, y_test = train_test_split(
    X_final, final_labels, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")

# -----------------------------
# 11. Baseline (Full Data)
# -----------------------------
X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

baseline_model = LogisticRegression(max_iter=200)
baseline_model.fit(X_full_train, y_full_train)
baseline_preds = baseline_model.predict(X_full_test)

print("Baseline Accuracy:", accuracy_score(y_full_test, baseline_preds))
