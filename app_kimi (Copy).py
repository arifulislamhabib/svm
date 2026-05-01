"""
=============================================================================
DATASET CONDENSATION PIPELINE
SVM-Based Support Vector Extraction + Smart Sampling
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.datasets import make_classification, load_breast_cancer, load_digits
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import entropy
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# STEP 0: DATASET LOADING (Choose one)
# =============================================================================

def load_synthetic_dataset(n_samples=10000, n_features=20, n_classes=2, 
                           n_informative=15, n_redundant=5, random_state=42):
    """Generate synthetic classification dataset"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=2,
        weights=[0.7, 0.3] if n_classes == 2 else None,
        flip_y=0.05,  # 5% label noise
        random_state=random_state
    )
    return X, y, "Synthetic_10K"

def load_breast_cancer_dataset():
    """Load breast cancer dataset"""
    data = load_breast_cancer()
    X, y = data.data, data.target
    return X, y, "Breast_Cancer"

def load_digits_dataset():
    """Load digits dataset"""
    data = load_digits()
    X, y = data.data, data.target
    return X, y, "Digits"

# =============================================================================
# STEP 1: INITIAL SVM TRAINING
# =============================================================================

class SVMTrainer:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """Train SVM and return model + support vectors"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        # Extract support vectors
        support_indices = self.model.support_
        support_vectors = X[support_indices]
        support_labels = y[support_indices]
        dual_coefs = np.abs(self.model.dual_coef_).flatten()
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'support_indices': support_indices,
            'support_vectors': support_vectors,
            'support_labels': support_labels,
            'dual_coefs': dual_coefs,
            'n_support': len(support_indices),
            'X_scaled': X_scaled
        }
    
    def predict(self, X):
        """Predict using trained SVM"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def decision_function(self, X):
        """Get decision function values (distance from hyperplane)"""
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)

# =============================================================================
# STEP 2: SV VALIDATION & PRUNING
# =============================================================================

class SVPruner:
    def __init__(self):
        self.clean_svs = None
        self.clean_labels = None
        
    # --- 2.1 Cross-Kernel Consensus SV ---
    def cross_kernel_consensus(self, X, y, support_indices, 
                               kernels=['rbf', 'poly', 'sigmoid'],
                               consensus_threshold=0.7):
        """
        Find SVs that are consistently important across different kernels
        """
        print("\n[Cross-Kernel Consensus SV]")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        
        sv_votes = np.zeros(len(support_indices))
        kernel_scores = {}
        
        for kernel in kernels:
            svm = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
            svm.fit(X_train_s, y_train)
            
            # Check which original SVs are also SVs in this kernel
            current_support = set(svm.support_)
            original_support = set(support_indices)
            
            # Map validation indices to original indices
            val_acc = accuracy_score(y_val, svm.predict(X_val_s))
            kernel_scores[kernel] = val_acc
            print(f"  {kernel}: Val Acc = {val_acc:.4f}")
            
            for i, idx in enumerate(support_indices):
                if idx in current_support:
                    sv_votes[i] += 1
        
        # Normalize votes
        sv_votes /= len(kernels)
        
        # Keep SVs with consensus >= threshold
        consensus_mask = sv_votes >= consensus_threshold
        consensus_indices = support_indices[consensus_mask]
        
        print(f"  Consensus threshold: {consensus_threshold}")
        print(f"  SVs before: {len(support_indices)}, After: {len(consensus_indices)}")
        
        return consensus_indices, sv_votes
    
    # --- 2.2 Noise-Aware SV Pruning ---
    def noise_aware_pruning(self, X, y, support_indices, 
                           k_neighbors=5, noise_threshold=0.3):
        """
        Remove SVs that are likely noise/outliers based on local neighborhood
        """
        print("\n[Noise-Aware SV Pruning]")
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        sv_data = X_s[support_indices]
        sv_labels = y[support_indices]
        
        # Compute local label consistency
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(X_s)
        distances, indices = nbrs.kneighbors(sv_data)
        
        noise_scores = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            # Exclude self
            neighbor_idx = idx[1:]
            neighbor_labels = y[neighbor_idx]
            
            # Label consistency ratio
            label_consistency = np.mean(neighbor_labels == sv_labels[i])
            noise_scores.append(1 - label_consistency)
        
        noise_scores = np.array(noise_scores)
        
        # Keep SVs with low noise score
        clean_mask = noise_scores <= noise_threshold
        clean_indices = support_indices[clean_mask]
        
        print(f"  Noise threshold: {noise_threshold}")
        print(f"  SVs before: {len(support_indices)}, After: {len(clean_indices)}")
        print(f"  Removed {np.sum(~clean_mask)} noisy SVs")
        
        return clean_indices, noise_scores
    
    # --- 2.3 Stability-Guided SV Selection ---
    def stability_guided_selection(self, X, y, support_indices, 
                                   n_bootstrap=10, stability_threshold=0.6):
        """
        Select SVs that appear consistently across bootstrap samples
        """
        print("\n[Stability-Guided SV Selection]")
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        n_samples = len(X)
        sv_stability = np.zeros(len(support_indices))
        
        for b in range(n_bootstrap):
            # Bootstrap sample
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_s[boot_idx]
            y_boot = y[boot_idx]
            
            # Train SVM on bootstrap
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=b)
            svm.fit(X_boot, y_boot)
            
            # Check which support indices appear
            boot_support = set(svm.support_)
            for i, idx in enumerate(support_indices):
                if idx in boot_idx:  # Must be in bootstrap
                    boot_local_idx = np.where(boot_idx == idx)[0]
                    if len(boot_local_idx) > 0:
                        local_idx = boot_local_idx[0]
                        if local_idx in boot_support:
                            sv_stability[i] += 1
        
        sv_stability /= n_bootstrap
        
        # Keep stable SVs
        stable_mask = sv_stability >= stability_threshold
        stable_indices = support_indices[stable_mask]
        
        print(f"  Bootstrap rounds: {n_bootstrap}")
        print(f"  Stability threshold: {stability_threshold}")
        print(f"  SVs before: {len(support_indices)}, After: {len(stable_indices)}")
        print(f"  Mean stability: {np.mean(sv_stability):.3f}")
        
        return stable_indices, sv_stability
    
    # --- Combined Pruning Pipeline ---
    def prune_support_vectors(self, X, y, svm_result, 
                            use_consensus=True,
                            use_noise=True,
                            use_stability=True):
        """
        Apply all pruning methods sequentially
        """
        support_indices = svm_result['support_indices']
        
        print("=" * 60)
        print("STEP 2: SV VALIDATION & PRUNING")
        print("=" * 60)
        
        current_indices = support_indices.copy()
        
        if use_consensus:
            current_indices, _ = self.cross_kernel_consensus(
                X, y, current_indices
            )
        
        if use_noise:
            current_indices, _ = self.noise_aware_pruning(
                X, y, current_indices
            )
        
        if use_stability:
            current_indices, _ = self.stability_guided_selection(
                X, y, current_indices
            )
        
        self.clean_svs = X[current_indices]
        self.clean_labels = y[current_indices]
        
        print(f"\n[FINAL] Clean SVs: {len(current_indices)} / {len(support_indices)} "
              f"({len(current_indices)/len(support_indices)*100:.1f}%)")
        
        return current_indices

# =============================================================================
# STEP 3: SMART SAMPLING FROM REMAINDER
# =============================================================================

class SmartSampler:
    def __init__(self):
        self.sampled_data = None
        self.sampled_labels = None
        
    # --- 3.1 Uncertainty-Weighted Smart Sampling ---
    def uncertainty_weighted_sampling(self, X_remain, y_remain, svm_trainer, 
                                     n_samples, alpha=1.0):
        """
        Sample points with highest uncertainty (closest to decision boundary)
        """
        print("\n[Uncertainty-Weighted Smart Sampling]")
        
        # Get decision function values
        decisions = svm_trainer.decision_function(X_remain)
        
        # Uncertainty = closeness to boundary (small absolute value)
        uncertainties = 1 / (np.abs(decisions) + 1e-10)
        uncertainties = uncertainties ** alpha
        
        # Normalize to probabilities
        probs = uncertainties / np.sum(uncertainties)
        
        # Weighted sampling without replacement
        n_select = min(n_samples, len(X_remain))
        selected_idx = np.random.choice(
            len(X_remain), size=n_select, replace=False, p=probs
        )
        
        print(f"  Requested: {n_samples}, Selected: {n_select}")
        print(f"  Mean uncertainty of selected: {np.mean(uncertainties[selected_idx]):.4f}")
        print(f"  Mean uncertainty of all: {np.mean(uncertainties):.4f}")
        
        return selected_idx
    
    # --- 3.2 Information-Theoretic Boundary Entropy Sampling ---
    def entropy_boundary_sampling(self, X_remain, y_remain, svm_trainer,
                                   n_samples, n_bins=10):
        """
        Sample points with high entropy near decision boundary
        """
        print("\n[Information-Theoretic Boundary Entropy Sampling]")
        
        # Get probability estimates
        X_scaled = svm_trainer.scaler.transform(X_remain)
        proba = svm_trainer.model.predict_proba(X_scaled)
        
        # Compute entropy for each point
        point_entropies = np.array([entropy(p) for p in proba])
        
        # Get decision values for boundary proximity
        decisions = np.abs(svm_trainer.decision_function(X_remain))
        
        # Combined score: high entropy + close to boundary
        # Normalize decisions
        norm_decisions = (decisions - decisions.min()) / (decisions.max() - decisions.min() + 1e-10)
        combined_score = point_entropies * (1 - norm_decisions + 0.1)
        
        # Select top scoring points
        n_select = min(n_samples, len(X_remain))
        selected_idx = np.argsort(combined_score)[-n_select:]
        
        print(f"  Selected: {n_select}")
        print(f"  Mean entropy selected: {np.mean(point_entropies[selected_idx]):.4f}")
        
        return selected_idx
    
    # --- 3.3 Sensitivity-Guided Sampling ---
    def sensitivity_guided_sampling(self, X_remain, y_remain, svm_trainer,
                                   n_samples, perturbation=0.01):
        """
        Sample points where small perturbations cause largest decision change
        """
        print("\n[Sensitivity-Guided Sampling]")
        
        X_scaled = svm_trainer.scaler.transform(X_remain)
        original_decisions = svm_trainer.model.decision_function(X_scaled)
        
        sensitivities = []
        for i in range(len(X_remain)):
            # Perturb point
            X_perturbed = X_scaled[i] + np.random.normal(0, perturbation, X_scaled[i].shape)
            X_perturbed = X_perturbed.reshape(1, -1)
            new_decision = svm_trainer.model.decision_function(X_perturbed)[0]
            
            sensitivity = np.abs(original_decisions[i] - new_decision)
            sensitivities.append(sensitivity)
        
        sensitivities = np.array(sensitivities)
        
        # Select most sensitive points
        n_select = min(n_samples, len(X_remain))
        selected_idx = np.argsort(sensitivities)[-n_select:]
        
        print(f"  Perturbation scale: {perturbation}")
        print(f"  Selected: {n_select}")
        print(f"  Mean sensitivity selected: {np.mean(sensitivities[selected_idx]):.4f}")
        
        return selected_idx
    
    # --- Combined Smart Sampling ---
    def sample_remainder(self, X, y, svm_trainer, clean_sv_indices, 
                         target_size, 
                         method='uncertainty',
                         ratio_uncertainty=0.5,
                         ratio_entropy=0.3,
                         ratio_sensitivity=0.2):
        """
        Smart sampling from remaining data
        """
        print("=" * 60)
        print("STEP 3: SMART SAMPLING FROM REMAINDER")
        print("=" * 60)
        
        # Get remaining data
        all_indices = np.arange(len(X))
        remain_mask = np.ones(len(X), dtype=bool)
        remain_mask[clean_sv_indices] = False
        remain_indices = all_indices[remain_mask]
        
        X_remain = X[remain_indices]
        y_remain = y[remain_mask]
        
        print(f"Remaining data: {len(X_remain)} samples")
        print(f"Target smart samples: {target_size}")
        
        if method == 'uncertainty':
            selected_local = self.uncertainty_weighted_sampling(
                X_remain, y_remain, svm_trainer, target_size
            )
        elif method == 'entropy':
            selected_local = self.entropy_boundary_sampling(
                X_remain, y_remain, svm_trainer, target_size
            )
        elif method == 'sensitivity':
            selected_local = self.sensitivity_guided_sampling(
                X_remain, y_remain, svm_trainer, target_size
            )
        elif method == 'hybrid':
            # Hybrid: combine all three
            n_unc = int(target_size * ratio_uncertainty)
            n_ent = int(target_size * ratio_entropy)
            n_sens = target_size - n_unc - n_ent
            
            idx_unc = self.uncertainty_weighted_sampling(
                X_remain, y_remain, svm_trainer, n_unc
            )
            idx_ent = self.entropy_boundary_sampling(
                X_remain, y_remain, svm_trainer, n_ent
            )
            idx_sens = self.sensitivity_guided_sampling(
                X_remain, y_remain, svm_trainer, n_sens
            )
            
            selected_local = np.unique(np.concatenate([idx_unc, idx_ent, idx_sens]))
            # If duplicates reduced size, fill with uncertainty
            if len(selected_local) < target_size:
                remaining = set(range(len(X_remain))) - set(selected_local)
                additional = np.random.choice(
                    list(remaining), 
                    size=min(target_size - len(selected_local), len(remaining)),
                    replace=False
                )
                selected_local = np.concatenate([selected_local, additional])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Map back to original indices
        selected_global = remain_indices[selected_local]
        
        self.sampled_data = X[selected_global]
        self.sampled_labels = y[selected_global]
        
        print(f"\n[FINAL] Smart samples: {len(selected_global)}")
        
        return selected_global

# =============================================================================
# STEP 4: MERGE WITH INTELLIGENT WEIGHTING
# =============================================================================

class DatasetMerger:
    def __init__(self):
        self.reduced_dataset = None
        self.reduced_labels = None
        self.weights = None
        
    # --- 4.1 Dual-Importance Distillation ---
    def dual_importance_weighting(self, clean_svs, clean_labels, sv_dual_coefs,
                                   smart_samples, smart_labels,
                                   svm_trainer):
        """
        Combine SVs and smart samples with dual importance weighting
        """
        print("\n[Dual-Importance Distillation]")
        
        # SV importance: from SVM dual coefficients
        sv_importance = sv_dual_coefs / (np.sum(sv_dual_coefs) + 1e-10)
        sv_importance = sv_importance / np.max(sv_importance)
        
        # Smart sample importance: uncertainty-based
        decisions = np.abs(svm_trainer.decision_function(smart_samples))
        smart_importance = 1 / (decisions + 1e-10)
        smart_importance = smart_importance / np.max(smart_importance)
        
        # Combine datasets
        X_combined = np.vstack([clean_svs, smart_samples])
        y_combined = np.concatenate([clean_labels, smart_labels])
        
        # Combine weights
        weights = np.concatenate([sv_importance, smart_importance])
        weights = weights / np.sum(weights) * len(weights)  # Normalize
        
        print(f"  SV importance range: [{sv_importance.min():.3f}, {sv_importance.max():.3f}]")
        print(f"  Smart sample importance range: [{smart_importance.min():.3f}, {smart_importance.max():.3f}]")
        
        return X_combined, y_combined, weights
    
    # --- 4.2 Information Decay Weighting ---
    def information_decay_weighting(self, X_combined, y_combined, base_weights,
                                     decay_factor=0.95):
        """
        Apply decay to older/less informative samples
        """
        print("\n[Information Decay Weighting]")
        
        # Sort by importance and apply decay
        sorted_idx = np.argsort(base_weights)[::-1]
        decay_weights = base_weights.copy()
        
        for rank, idx in enumerate(sorted_idx):
            decay_weights[idx] *= (decay_factor ** rank)
        
        # Renormalize
        decay_weights = decay_weights / np.sum(decay_weights) * len(decay_weights)
        
        print(f"  Decay factor: {decay_factor}")
        print(f"  Weight range before: [{base_weights.min():.3f}, {base_weights.max():.3f}]")
        print(f"  Weight range after: [{decay_weights.min():.3f}, {decay_weights.max():.3f}]")
        
        return decay_weights
    
    def merge_datasets(self, X, y, clean_sv_indices, smart_sample_indices,
                       svm_result, use_decay=True):
        """
        Merge clean SVs and smart samples with intelligent weighting
        """
        print("=" * 60)
        print("STEP 4: MERGE WITH INTELLIGENT WEIGHTING")
        print("=" * 60)
        
        clean_svs = X[clean_sv_indices]
        clean_labels = y[clean_sv_indices]
        
        # Get dual coefficients for clean SVs
        all_dual_coefs = svm_result['dual_coefs']
        # Map to clean indices
        sv_idx_map = {idx: i for i, idx in enumerate(svm_result['support_indices'])}
        clean_dual_coefs = np.array([
            all_dual_coefs[sv_idx_map[idx]] for idx in clean_sv_indices 
            if idx in sv_idx_map
        ])
        
        smart_samples = X[smart_sample_indices]
        smart_labels = y[smart_sample_indices]
        
        # Dual importance weighting
        X_combined, y_combined, weights = self.dual_importance_weighting(
            clean_svs, clean_labels, clean_dual_coefs,
            smart_samples, smart_labels,
            SVMTrainer()  # Create new for prediction
        )
        # Actually use the trained one
        from sklearn.svm import SVC
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        svm_model = SVC(kernel='rbf', probability=True)
        svm_model.fit(X_s, y)
        
        # Recreate with proper trainer
        trainer = SVMTrainer()
        trainer.model = svm_model
        trainer.scaler = scaler
        
        X_combined, y_combined, weights = self.dual_importance_weighting(
            clean_svs, clean_labels, clean_dual_coefs,
            smart_samples, smart_labels,
            trainer
        )
        
        if use_decay:
            weights = self.information_decay_weighting(X_combined, y_combined, weights)
        
        self.reduced_dataset = X_combined
        self.reduced_labels = y_combined
        self.weights = weights
        
        print(f"\n[FINAL] Reduced dataset: {len(X_combined)} samples")
        print(f"  - Clean SVs: {len(clean_svs)}")
        print(f"  - Smart samples: {len(smart_samples)}")
        print(f"  - Compression ratio: {len(X) / len(X_combined):.2f}x")
        
        return X_combined, y_combined, weights

# =============================================================================
# STEP 5: VALIDATION & COMPARISON
# =============================================================================

class ModelValidator:
    def __init__(self):
        self.results = {}
        
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, 
                           dataset_name, sample_weights=None):
        """
        Train multiple models and evaluate
        """
        print(f"\n{'='*60}")
        print(f"Training on: {dataset_name}")
        print(f"Samples: {len(X_train)}")
        print(f"{'='*60}")
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            print(f"\n  Training {name}...")
            
            if sample_weights is not None:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'accuracy': acc,
                'f1': f1,
                'model': model
            }
            
            print(f"    Accuracy: {acc:.4f}")
            print(f"    F1 Score: {f1:.4f}")
        
        self.results[dataset_name] = results
        return results
    
    def compare_datasets(self, X_full, y_full, X_reduced, y_reduced, 
                         weights_reduced, test_size=0.2):
        """
        Compare full vs reduced dataset performance
        """
        print("\n" + "="*70)
        print("STEP 5: VALIDATION - FULL vs REDUCED DATASET")
        print("="*70)
        
        # Split full dataset
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_full, y_full, test_size=test_size, random_state=42, stratify=y_full
        )
        
        # Use same test set for reduced
        # Scale data
        scaler_full = StandardScaler()
        X_train_full_s = scaler_full.fit_transform(X_train_full)
        X_test_s = scaler_full.transform(X_test)
        
        scaler_red = StandardScaler()
        X_train_red_s = scaler_red.fit_transform(X_reduced)
        X_test_red_s = scaler_red.transform(X_test)
        
        # Train on full
        results_full = self.train_and_evaluate(
            X_train_full_s, y_train_full, X_test_s, y_test, "FULL_DATASET"
        )
        
        # Train on reduced
        results_reduced = self.train_and_evaluate(
            X_train_red_s, y_reduced, X_test_red_s, y_test, 
            "REDUCED_DATASET",
            sample_weights=weights_reduced if weights_reduced is not None else None
        )
        
        # Comparison
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Model':<20} {'Full Acc':<12} {'Reduced Acc':<12} {'Retention':<12}")
        print("-"*70)
        
        for model_name in results_full.keys():
            full_acc = results_full[model_name]['accuracy']
            red_acc = results_reduced[model_name]['accuracy']
            retention = (red_acc / full_acc) * 100 if full_acc > 0 else 0
            
            print(f"{model_name:<20} {full_acc:<12.4f} {red_acc:<12.4f} {retention:<12.1f}%")
        
        # Overall statistics
        avg_full = np.mean([r['accuracy'] for r in results_full.values()])
        avg_red = np.mean([r['accuracy'] for r in results_reduced.values()])
        
        print(f"\nAverage Accuracy Full:    {avg_full:.4f}")
        print(f"Average Accuracy Reduced: {avg_red:.4f}")
        print(f"Overall Retention:        {(avg_red/avg_full)*100:.2f}%")
        print(f"Compression Ratio:        {len(X_full)/len(X_reduced):.2f}x")
        
        return results_full, results_reduced

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_dataset_condensation_pipeline(X, y, dataset_name,
                                      target_reduction=0.1,  # Keep 10% of data
                                      sampling_method='hybrid'):
    """
    Run complete dataset condensation pipeline
    """
    print("\n" + "="*70)
    print(f"DATASET CONDENSATION PIPELINE: {dataset_name}")
    print(f"Original size: {len(X)} samples, {X.shape[1]} features")
    print(f"Target reduction: {target_reduction*100}%")
    print("="*70)
    
    # Step 1: Train SVM
    print("\n" + "="*60)
    print("STEP 1: INITIAL SVM TRAINING")
    print("="*60)
    
    svm_trainer = SVMTrainer(kernel='rbf', C=1.0, gamma='scale')
    svm_result = svm_trainer.fit(X, y)
    
    print(f"Total SVs extracted: {svm_result['n_support']}")
    print(f"SV ratio: {svm_result['n_support']/len(X)*100:.2f}%")
    
    # Step 2: Prune SVs
    pruner = SVPruner()
    clean_sv_indices = pruner.prune_support_vectors(X, y, svm_result)
    
    # Step 3: Smart Sampling
    target_smart = int(len(X) * target_reduction) - len(clean_sv_indices)
    target_smart = max(target_smart, len(clean_sv_indices))  # At least as many as SVs
    
    sampler = SmartSampler()
    smart_indices = sampler.sample_remainder(
        X, y, svm_trainer, clean_sv_indices, target_smart, method=sampling_method
    )
    
    # Step 4: Merge
    merger = DatasetMerger()
    X_reduced, y_reduced, weights = merger.merge_datasets(
        X, y, clean_sv_indices, smart_indices, svm_result
    )
    
    # Step 5: Validate
    validator = ModelValidator()
    results_full, results_reduced = validator.compare_datasets(
        X, y, X_reduced, y_reduced, weights
    )
    
    return {
        'original_size': len(X),
        'reduced_size': len(X_reduced),
        'compression_ratio': len(X) / len(X_reduced),
        'reduced_dataset': (X_reduced, y_reduced, weights),
        'results': (results_full, results_reduced)
    }

# =============================================================================
# RUN EXAMPLES
# =============================================================================

if __name__ == "__main__":
    # Example 1: Synthetic Dataset
    print("\n" + "#"*70)
    print("EXAMPLE 1: SYNTHETIC DATASET")
    print("#"*70)
    
    X_syn, y_syn, name_syn = load_synthetic_dataset(n_samples=10000)
    result_syn = run_dataset_condensation_pipeline(
        X_syn, y_syn, name_syn, target_reduction=0.1, sampling_method='hybrid'
    )
    
    # Example 2: Breast Cancer
    print("\n" + "#"*70)
    print("EXAMPLE 2: BREAST CANCER DATASET")
    print("#"*70)
    
    X_bc, y_bc, name_bc = load_breast_cancer_dataset()
    result_bc = run_dataset_condensation_pipeline(
        X_bc, y_bc, name_bc, target_reduction=0.2, sampling_method='uncertainty'
    )
    
    # Example 3: Digits
    print("\n" + "#"*70)
    print("EXAMPLE 3: DIGITS DATASET")
    print("#"*70)
    
    X_dig, y_dig, name_dig = load_digits_dataset()
    result_dig = run_dataset_condensation_pipeline(
        X_dig, y_dig, name_dig, target_reduction=0.15, sampling_method='hybrid'
    )
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY ACROSS ALL DATASETS")
    print("="*70)
    
    for name, result in [("Synthetic", result_syn), 
                          ("Breast Cancer", result_bc), 
                          ("Digits", result_dig)]:
        print(f"\n{name}:")
        print(f"  Original: {result['original_size']}, Reduced: {result['reduced_size']}")
        print(f"  Compression: {result['compression_ratio']:.2f}x")
