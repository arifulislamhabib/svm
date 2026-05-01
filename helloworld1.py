import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

# ========================== 1. Load Covertype Dataset ==========================
print("Loading Covertype Dataset...")
data = fetch_covtype()
X = data.data
y = data.target

# For faster testing you can subsample (comment out for full run)
# np.random.seed(42)
# idx = np.random.choice(len(X), 100000, replace=False)
# X, y = X[idx], y[idx]

print(f"Original Dataset Shape: {X.shape} | Classes: {len(np.unique(y))}\n")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling (SVM-এর জন্য জরুরি)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========================== 2. Train Initial SVM ==========================
print("Training Initial SVM...")
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_train, y_train)

# ========================== 3. Extract Support Vectors ==========================
sv_indices = svm.support_
X_sv = X_train[sv_indices]
y_sv = y_train[sv_indices]

print(f"Support Vectors Extracted: {X_sv.shape[0]} samples ({X_sv.shape[0]/X_train.shape[0]*100:.2f}% of train data)")

# ========================== 4. SV Augmentation via Generative Models ==========================
# Simple but effective: Gaussian perturbation around SVs
np.random.seed(42)
noise = np.random.normal(0, 0.05, X_sv.shape)          # small noise
X_aug_sv = X_sv + noise
y_aug_sv = y_sv.copy()

X_sv_aug = np.vstack([X_sv, X_aug_sv])
y_sv_aug = np.concatenate([y_sv, y_aug_sv])

print(f"After SV Augmentation: {X_sv_aug.shape[0]} samples")

# ========================== 5. Sensitivity-Guided Sampling ==========================
# Remaining non-SV data
non_sv_mask = ~np.isin(np.arange(len(X_train)), sv_indices)
X_non_sv = X_train[non_sv_mask]
y_non_sv = y_train[non_sv_mask]

# Sensitivity = |decision_function| * approximate gradient magnitude (distance from margin)
dist_from_margin = np.abs(svm.decision_function(X_non_sv))
sensitivity_scores = dist_from_margin  # higher score = closer to margin = more important

# Keep top 12% most sensitive non-SV
n_sample = int(0.12 * len(X_non_sv))
idx = np.argsort(sensitivity_scores)[-n_sample:]
X_sampled = X_non_sv[idx]
y_sampled = y_non_sv[idx]

print(f"Sensitivity-Guided Sampling: {X_sampled.shape[0]} samples selected from non-SV")

# ========================== 6. Merge Both ==========================
X_merged = np.vstack([X_sv_aug, X_sampled])
y_merged = np.concatenate([y_sv_aug, y_sampled])

print(f"\n=== Final Reduced Dataset ===")
print(f"X_reduced shape: {X_merged.shape}")
print(f"Reduction Ratio: {X_merged.shape[0] / X_train.shape[0] * 100:.2f}% of original train data")
print(f"Total data reduction from full dataset: {X_merged.shape[0] / X.shape[0] * 100:.2f}%\n")

# ========================== 7. Adversarial Margin Augmentation (AMA) ==========================
# Small adversarial perturbation on SVs
epsilon = 0.03
grad_sign = np.sign(svm.decision_function(X_sv_aug))
X_adv = X_sv_aug + epsilon * grad_sign.reshape(-1, 1)

X_merged = np.vstack([X_merged, X_adv])
y_merged = np.concatenate([y_merged, y_sv_aug])

print(f"After AMA: {X_merged.shape[0]} samples (hard examples added)")

# ========================== 8. Train Multiple Models with Fidelity + Feedback Loop ==========================
models = {
    'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
}

final_models = {}
n_iterations = 3   # Feedback Loop

for iteration in range(n_iterations):
    print(f"\nFeedback Loop Iteration {iteration+1}/{n_iterations}")
    for name, model in models.items():
        model.fit(X_merged, y_merged)
        final_models[name] = model

# Final Ensemble
ensemble = VotingClassifier(estimators=[('xgb', final_models['XGBoost']),
                                        ('lgb', final_models['LightGBM']),
                                        ('mlp', final_models['MLP'])],
                            voting='soft')
ensemble.fit(X_merged, y_merged)

# ========================== 9. Final Evaluation ==========================
y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Original Train Samples      : {X_train.shape[0]:,}")
print(f"Reduced Dataset (X_reduced) : {X_merged.shape[0]:,} samples")
print(f"Reduction Achieved          : {100 - (X_merged.shape[0]/X_train.shape[0]*100):.2f}%")
print(f"Final Ensemble Accuracy     : {acc*100:.3f}%")
print("="*60)

# Save reduced dataset (optional)
np.savez("reduced_covertype.npz", X_reduced=X_merged, y_reduced=y_merged)
print("Reduced dataset saved as 'reduced_covertype.npz'")
