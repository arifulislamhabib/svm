"""
Complete implementation of:
SVM → Support Vectors → Smart Sampling (K‑means) → Merge → Multiple Models
→ Ensemble Disagreement Feedback Loop
Dataset: Forest CoverType (binary classification)
"""

import numpy as np
import time
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 1. Load and prepare dataset
# ===============================
print("Loading CoverType dataset...")
data = fetch_covtype()
X = data.data
y = data.target

# Convert to binary: class 1 vs all others (1 = Pass, -1 = Fail)
y = np.where(y == 1, 1, -1)

# Split into train (70%), validation (15%), test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)  # 0.1765 of 0.85 ≈ 0.15

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

# ===============================
# 2. Step 1: Train SVM on full training set
# ===============================
print("\n--- Step 1: Training SVM (may take a few minutes) ---")
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
start_svm = time.time()
svm.fit(X_train, y_train)
svm_time = time.time() - start_svm
sv_indices = svm.support_
X_sv = X_train[sv_indices]
y_sv = y_train[sv_indices]
print(f"SVM trained in {svm_time:.2f}s. Support vectors: {len(X_sv)}/{len(X_train)}")

# ===============================
# 3. Step 2: Smart sampling (K‑means) on non‑SV points
# ===============================
print("\n--- Step 2: Smart sampling on non‑SV points ---")
all_indices = np.arange(len(X_train))
non_sv_indices = np.setdiff1d(all_indices, sv_indices)
X_non_sv = X_train[non_sv_indices]
y_non_sv = y_train[non_sv_indices]

# Sample same number as SVs (or 2x, etc.)
k_extra = len(X_sv)  # 1:1 ratio
k_extra = min(k_extra, len(X_non_sv))

# K‑means clustering on non‑SV points
kmeans = KMeans(n_clusters=k_extra, random_state=42, n_init='auto')
kmeans.fit(X_non_sv)

# For each cluster, pick point nearest to centroid
sampled_indices = []
for i in range(k_extra):
    mask = (kmeans.labels_ == i)
    cluster_points = X_non_sv[mask]
    if len(cluster_points) == 0:
        continue
    centroid = kmeans.cluster_centers_[i]
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    closest = np.argmin(distances)
    orig_idx = np.where(mask)[0][closest]
    sampled_indices.append(orig_idx)

X_sampled = X_non_sv[sampled_indices]
y_sampled = y_non_sv[sampled_indices]
print(f"Sampled {len(X_sampled)} points via K‑means")

# ===============================
# 4. Step 3: Merge
# ===============================
X_reduced = np.vstack([X_sv, X_sampled])
y_reduced = np.hstack([y_sv, y_sampled])
print(f"\n--- Step 3: Merged reduced dataset size = {len(X_reduced)} (reduction ratio: {len(X_train)/len(X_reduced):.1f}x)")

# ===============================
# 5. Step 4: Train multiple models (initial)
# ===============================
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
}

print("\n--- Step 4: Training models on reduced dataset ---")
for name, model in models.items():
    start = time.time()
    model.fit(X_reduced, y_reduced)
    elapsed = time.time() - start
    print(f"{name} trained in {elapsed:.2f}s")

# ===============================
# 6. Step 5: Feedback Loop (Ensemble Disagreement)
# ===============================
print("\n--- Step 5: Feedback Loop (add disagreement points) ---")
max_iterations = 5
iteration = 0
disagreement_counts = []

while iteration < max_iterations:
    # Get predictions on validation set from all models
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict(X_val)
    
    # Find indices where not all models agree
    disagree_idx = []
    for i in range(len(X_val)):
        pred_list = [preds[name][i] for name in models.keys()]
        if len(set(pred_list)) > 1:   # disagreement
            disagree_idx.append(i)
    
    n_disagree = len(disagree_idx)
    disagreement_counts.append(n_disagree)
    print(f"Iteration {iteration+1}: {n_disagree} disagreement points found")
    
    if n_disagree == 0:
        print("Converged! No disagreement.")
        break
    
    # Add disagreement points to reduced dataset
    X_disagree = X_val[disagree_idx]
    y_disagree = y_val[disagree_idx]
    X_reduced = np.vstack([X_reduced, X_disagree])
    y_reduced = np.hstack([y_reduced, y_disagree])
    print(f"Added {n_disagree} points. New reduced size: {len(X_reduced)}")
    
    # Retrain models on enlarged reduced dataset
    for name, model in models.items():
        model.fit(X_reduced, y_reduced)
    
    iteration += 1

# ===============================
# 7. Final evaluation on test set
# ===============================
print("\n--- Final Evaluation on Test Set ---")
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {acc:.4f} ({acc*100:.2f}%)")

print(f"\nFeedback loop converged after {iteration+1} iterations.")
print(f"Disagreement counts: {disagreement_counts}")
print(f"Final reduced dataset size: {len(X_reduced)} (from {len(X_train)} original)")
print(f"Final reduction ratio: {len(X_train)/len(X_reduced):.1f}x")
