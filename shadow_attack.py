import os
import json
import gc
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import numpy as np
from scipy.stats import norm
from scipy.stats import t
import torchvision
from global_variables import DATA_DIR, TRANSFORM_TRAIN, TRANSFORM_TEST
import torchvision.datasets as datasets
from sklearn.metrics import roc_curve


from resnet_influence import ResNet18_Influence
from train_utils import TrainingConfig, train_shadow_model

def evaluate_target_model(model_path, query_indices, device):
    print(f"Loading Target Model from {model_path}...")
    
    model = ResNet18_Influence()
    
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
        
    model.to(device)
    model.eval()
    

    full_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=TRANSFORM_TRAIN)
    
    query_dataset = Subset(full_dataset, query_indices)
    dataloader = DataLoader(query_dataset, batch_size=128, shuffle=False)
    
    target_scores = []
    
    print("Evaluating Query Points on Target Model...")
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            
            # Extract true class probability
            p_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            # CLAMP to prevent inf / -inf
            p_true = torch.clamp(p_true, min=1e-7, max=1.0 - 1e-7)
            
            # Compute LiRA score exactly as done in shadow models
            scores = torch.log(p_true / (1.0 - p_true)).cpu().numpy()
            target_scores.extend(scores)
            
    return np.array(target_scores)



def train_single_shadow_model(subset_indices, device, training_config, checkpoint_path=None):
    """Physically trains ONE ResNet on the specified subset of CIFAR-10.
    
    Args:
        subset_indices: List of training sample indices
        device: torch device
        training_config: TrainingConfig object from target model metadata
        checkpoint_path: Optional path to save/resume checkpoint
    
    Returns:
        Trained model in eval mode
    """
    model = ResNet18_Influence().to(device)
    trained_model = train_shadow_model(model, subset_indices, training_config, device, checkpoint_path)
    return trained_model


def _shadow_cache_metadata_path(shadow_models_dir):
    return os.path.join(shadow_models_dir, 'cache_metadata.npz')


def _normalized_confident_memberships(confident_memberships, num_queries):
    if confident_memberships is None:
        return np.full(num_queries, -1, dtype=np.int8)
    return np.asarray(confident_memberships, dtype=np.int8)


def _shadow_cache_matches(metadata_path, query_indices, confident_memberships, num_models, target_train_size):
    if not os.path.exists(metadata_path):
        return False, 'cache metadata is missing'

    cache_metadata = np.load(metadata_path)
    cached_query_indices = cache_metadata['query_indices']
    cached_confident_memberships = cache_metadata['confident_memberships']
    cached_num_models = int(cache_metadata['num_models'][0])
    cached_target_train_size = int(cache_metadata['target_train_size'][0])

    expected_query_indices = np.asarray(query_indices, dtype=np.int64)
    expected_confident_memberships = _normalized_confident_memberships(
        confident_memberships,
        len(query_indices)
    )

    if cached_num_models != num_models:
        return False, f'cached num_models={cached_num_models} does not match requested {num_models}'
    if cached_target_train_size != target_train_size:
        return False, (
            f'cached target_train_size={cached_target_train_size} does not match '
            f'requested {target_train_size}'
        )
    if not np.array_equal(cached_query_indices, expected_query_indices):
        return False, 'cached query indices do not match the current attack run'
    if not np.array_equal(cached_confident_memberships, expected_confident_memberships):
        return False, 'cached anchors do not match the current run'

    return True, None


def _write_shadow_cache_metadata(shadow_models_dir, query_indices, confident_memberships, num_models, target_train_size):
    np.savez(
        _shadow_cache_metadata_path(shadow_models_dir),
        query_indices=np.asarray(query_indices, dtype=np.int64),
        confident_memberships=_normalized_confident_memberships(
            confident_memberships,
            len(query_indices)
        ),
        num_models=np.array([num_models], dtype=np.int64),
        target_train_size=np.array([target_train_size], dtype=np.int64),
    )

def train_shadow_models(query_indices, num_models, total_dataset_size, target_train_size, 
                       training_config, confident_memberships=None, shadow_models_dir=None):
    """
    Train shadow models using the same configuration as the target model.
    Supports resuming from partially completed training runs.
    
    Args:
        query_indices: Indices of query points
        num_models: Number of shadow models to train
        total_dataset_size: Total size of CIFAR-10 training set
        target_train_size: Size of target model's training set
        training_config: TrainingConfig object from target model metadata
        confident_memberships: Optional array of anchored memberships (-1 for unknown, 0/1 for known)
        shadow_models_dir: Directory to save shadow models (enables checkpointing if provided)
    
    Returns:
        Tuple of (shadow_models, shadow_datasets_m, shadow_subsets, any_new_models_trained)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shadow_models = []
    shadow_datasets_m = [] 
    shadow_subsets = []
    any_new_models_trained = False
    
    all_indices = set(range(total_dataset_size))
    background_pool = list(all_indices - set(query_indices))
    
    # Check if we're resuming from a previous run
    membership_matrix_path = None
    cache_metadata_path = None
    allow_cache_reuse = True
    if shadow_models_dir is not None:
        os.makedirs(shadow_models_dir, exist_ok=True)
        membership_matrix_path = os.path.join(shadow_models_dir, 'membership_matrix.npy')
        cache_metadata_path = _shadow_cache_metadata_path(shadow_models_dir)

        shadow_model_paths = [
            os.path.join(shadow_models_dir, f"shadow_{k}.pth")
            for k in range(num_models)
        ]
        has_existing_cache = os.path.exists(membership_matrix_path) or any(
            os.path.exists(path) for path in shadow_model_paths
        )
        if has_existing_cache:
            cache_matches, mismatch_reason = _shadow_cache_matches(
                cache_metadata_path,
                query_indices,
                confident_memberships,
                num_models,
                target_train_size,
            )
            if not cache_matches:
                print(
                    f"  -> Ignoring existing shadow cache in {shadow_models_dir}: {mismatch_reason}."
                )
                allow_cache_reuse = False

        _write_shadow_cache_metadata(
            shadow_models_dir,
            query_indices,
            confident_memberships,
            num_models,
            target_train_size,
        )
    
    # Load or create membership matrix
    if allow_cache_reuse and membership_matrix_path and os.path.exists(membership_matrix_path):
        print(f"  -> Loading existing membership matrix from {membership_matrix_path}")
        existing_matrix = np.load(membership_matrix_path)
        existing_num_models = existing_matrix.shape[0]
        
        if existing_num_models == num_models:
            # Perfect match, use as is
            membership_matrix = existing_matrix
        elif existing_num_models < num_models:
            # Need to extend the matrix with additional models
            print(f"  -> Extending membership matrix from {existing_num_models} to {num_models} models")
            membership_matrix = np.zeros((num_models, len(query_indices)), dtype=int)
            membership_matrix[:existing_num_models, :] = existing_matrix
            
            # For each query point, compute how many additional IN assignments are needed
            # to maintain ~50% IN across all num_models
            for i, idx in enumerate(query_indices):
                if confident_memberships is not None and confident_memberships[i] != -1:
                    # Anchored point - set all new models to the anchored value
                    membership_matrix[existing_num_models:, i] = confident_memberships[i]
                else:
                    # Count how many existing models have this query point as IN
                    existing_in_count = np.sum(existing_matrix[:, i])
                    # Target is roughly half of all models
                    target_in_count = num_models // 2
                    # How many more models should have this as IN?
                    additional_in_needed = max(0, target_in_count - existing_in_count)
                    
                    # Randomly select from the new models to add IN
                    num_new_models = num_models - existing_num_models
                    additional_in_needed = min(additional_in_needed, num_new_models)
                    
                    if additional_in_needed > 0:
                        new_models_with_in = np.random.choice(
                            num_new_models, 
                            size=additional_in_needed, 
                            replace=False
                        ) + existing_num_models
                        membership_matrix[new_models_with_in, i] = 1
            
            # Save the extended matrix
            np.save(membership_matrix_path, membership_matrix)
            print(f"  -> Saved extended membership matrix to {membership_matrix_path}")
        else:
            # More models in file than requested - use first num_models
            print(f"  -> WARNING: Existing matrix has {existing_num_models} models but only {num_models} requested")
            print(f"  -> Using first {num_models} models from existing matrix")
            membership_matrix = existing_matrix[:num_models, :]
    else:
        # Pre-compute membership matrix to ensure proper distribution
        # Each query point should be IN for exactly half the models
        membership_matrix = np.zeros((num_models, len(query_indices)), dtype=int)
        
        for i, idx in enumerate(query_indices):
            if confident_memberships is not None and confident_memberships[i] != -1:
                # If this query point is anchored, all models must respect it
                membership_matrix[:, i] = confident_memberships[i]
            else:
                # Randomly select half the models to have this query point as IN
                models_with_in = np.random.choice(num_models, size=num_models // 2, replace=False)
                membership_matrix[models_with_in, i] = 1
        
        # Save membership matrix for resumability
        if membership_matrix_path:
            np.save(membership_matrix_path, membership_matrix)
            print(f"  -> Saved membership matrix to {membership_matrix_path}")
    
    # Now train each shadow model based on the pre-computed membership matrix
    for k in range(num_models):
        # Check if this shadow model already exists
        if shadow_models_dir is not None and allow_cache_reuse:
            shadow_model_path = os.path.join(shadow_models_dir, f"shadow_{k}.pth")
            if os.path.exists(shadow_model_path):
                print(f"  -> Loading existing Shadow Model {k+1}/{num_models} from {shadow_model_path}")
                # Load the model
                model = ResNet18_Influence()
                checkpoint = torch.load(shadow_model_path, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                shadow_models.append(model)
                shadow_datasets_m.append(checkpoint['m_k'])
                shadow_subsets.append(checkpoint['subset_indices'])
                continue
        
        print(f"  -> Training Shadow Model {k+1}/{num_models}...")
        any_new_models_trained = True
        
        m_k = membership_matrix[k]
        subset_indices = []
        
        # Add query points that are IN for this model
        for i, idx in enumerate(query_indices):
            if m_k[i] == 1:
                subset_indices.append(int(idx))
                
        num_background_needed = target_train_size - len(subset_indices)
        
        # Deterministic seed per model to ensure reproducibility
        # This ensures model k always gets the same background samples
        rng = np.random.RandomState(seed=1000 + k)
        background_sample = rng.choice(background_pool, num_background_needed, replace=False)
        subset_indices.extend(background_sample.tolist())
        
        shadow_subsets.append(subset_indices)
        
        # Train with checkpointing support
        checkpoint_path = None
        if shadow_models_dir is not None:
            checkpoint_path = os.path.join(shadow_models_dir, f"shadow_{k}.pth")
        
        trained_model = train_single_shadow_model(subset_indices, device, training_config, checkpoint_path)
        
        shadow_models.append(trained_model.cpu()) # this makes copy of model in CPU
        shadow_datasets_m.append(m_k)
        
        # Save the completed shadow model immediately
        if shadow_models_dir is not None:
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'm_k': m_k,
                'subset_indices': subset_indices
            }, checkpoint_path)
            print(f"     Saved shadow model {k+1} to {checkpoint_path}")
        
        del trained_model 
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return shadow_models, np.array(shadow_datasets_m), shadow_subsets, any_new_models_trained


def precompute_influence_matrices(shadow_models, shadow_subsets, query_indices, device, hessian_sample_size=3000):
    """
    Calculate Hessian and Influence matrix C using a subsampled Hessian for speed.
    Note: shadow_subsets MUST be the list of exact index arrays each model was trained on.
    """
    C_matrices = []
    t_bases = []
    
    full_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=TRANSFORM_TEST)
    
    for k, model in enumerate(shadow_models):
        print(f"  -> Precomputing Influence for Shadow Model {k+1}/{len(shadow_models)}...")
        model = model.to(device)
        
        # 1. FAST HESSIAN APPROXIMATION
        train_indices = shadow_subsets[k]
        
        # Randomly subsample the training set to compute the Hessian quickly
        sample_size = min(hessian_sample_size, len(train_indices))
        hessian_indices = np.random.choice(train_indices, sample_size, replace=False)
        hessian_subset = Subset(full_dataset, hessian_indices)
        
        hessian_loader = DataLoader(hessian_subset, batch_size=256, shuffle=False, num_workers=0)
        
        # Compute H on the subsample
        H = model.compute_last_layer_hessian(hessian_loader, device)
        H_inv = torch.linalg.inv(H)
        
        # 2. INFLUENCE MATRIX FOR QUERY POINTS ONLY
        # This remains N x N just for the influence points!
        query_subset = Subset(full_dataset, [int(idx) for idx in query_indices])
        query_loader = DataLoader(query_subset, batch_size=256, shuffle=False)
        
        # Pass the TRUE train size (len(train_indices)), not the subsample size, 
        # to ensure the 1/N scaling is mathematically correct
        train_size = len(train_indices) 
        C = model.compute_influence_matrix(query_loader, H_inv, train_size,device)
        C_matrices.append(C.cpu().numpy())
        
        t_base = model.get_lira_statistics(query_loader, device) 
        t_bases.append(t_base.cpu().numpy())
        
        # memory clean up
        model = model.cpu()
        del H, H_inv, C
        torch.cuda.empty_cache()
    print("done precomputing influence matrices.")
    return np.array(C_matrices), np.array(t_bases)

def run_lira_baseline(target_scores, t_bases, m_actual, ground_truth, attack_dir):
    """
    Standard LiRA (Likelihood Ratio Attack) baseline — no MCMC, no influence functions.

    For each query point i:
      - Collect shadow logit scores where point was IN vs OUT
      - Fit Gaussian (mean, std) to each set
      - Score = log p(target_score | IN Gaussian) - log p(target_score | OUT Gaussian)
    A higher score means the model thinks the point is a member.

    Args:
        target_scores: Logit scores from the target model, shape (N,)
        t_bases:       Shadow model logit scores, shape (K, N)
        m_actual:      Membership matrix, shape (K, N), values 0 or 1
        ground_truth:  Ground truth membership, shape (N,)
        attack_dir:    Directory to save results

    Returns:
        lira_scores: Per-point LiRA log-likelihood ratio scores, shape (N,)
        tpr_at_01pct: TPR at 0.1% FPR
        tpr_at_1pct:  TPR at 1% FPR
    """
    N = len(target_scores)
    K = len(t_bases)
    global_std = np.std(t_bases, ddof=1) + 1e-8

    lira_scores = np.zeros(N, dtype=np.float64)

    for i in range(N):
        in_scores  = t_bases[m_actual[:, i] == 1, i]
        out_scores = t_bases[m_actual[:, i] == 0, i]

        mu_in  = np.mean(in_scores)  if len(in_scores)  > 0 else 0.0
        mu_out = np.mean(out_scores) if len(out_scores) > 0 else 0.0
        std_in  = np.std(in_scores,  ddof=1) if len(in_scores)  > 1 else global_std
        std_out = np.std(out_scores, ddof=1) if len(out_scores) > 1 else global_std
        std_in  = std_in  if std_in  > 0 else global_std
        std_out = std_out if std_out > 0 else global_std

        log_p_in  = norm.logpdf(target_scores[i], loc=mu_in,  scale=std_in)
        log_p_out = norm.logpdf(target_scores[i], loc=mu_out, scale=std_out)
        lira_scores[i] = log_p_in - log_p_out

    # Convert log-LR to a [0, 1] posterior-like probability via sigmoid for comparability
    lira_probs = 1.0 / (1.0 + np.exp(-lira_scores))

    # Save results
    lira_results = {str(int(i)): float(lira_probs[i]) for i in range(N)}
    with open(os.path.join(attack_dir, "lira_baseline_probs.json"), "w") as f:
        json.dump(lira_results, f, indent=2)
    np.save(os.path.join(attack_dir, "lira_baseline_scores.npy"), lira_scores)

    # Metrics
    fpr_curve, tpr_curve, _ = roc_curve(ground_truth, lira_probs)
    valid_01 = np.where(fpr_curve <= 0.001)[0]
    tpr_at_01pct = tpr_curve[valid_01[-1]] if len(valid_01) > 0 else float('nan')
    valid_1 = np.where(fpr_curve <= 0.01)[0]
    tpr_at_1pct = tpr_curve[valid_1[-1]] if len(valid_1) > 0 else float('nan')

    return lira_scores, tpr_at_01pct, tpr_at_1pct


def run_mcmc(target_scores, C_matrices, t_bases, m_actual, output_dir, num_steps=10000, prior_prob=0.5, temperature=0.7):
    N = len(target_scores)
    K = len(C_matrices)
    global_std = np.std(t_bases, ddof=1) + 1e-8
    
    # --- 1. PRECOMPUTE STATIC VARIABLES ONCE ---
    log_p = np.log(prior_prob + 1e-10)
    log_1_p = np.log(1 - prior_prob + 1e-10)
    trace_file_path = os.path.join(output_dir, "mcmc_trace.npy")
    valid_indices = np.arange(N)
        
    C_diag = np.diagonal(C_matrices, axis1=1, axis2=2)
    in_mask = (m_actual == 1)
    out_mask = (m_actual == 0)
    df = (K / 2) - 1
    
    # --- 2. INITIALIZE STATE ---
    M_current = np.random.randint(0, 2, size=N, dtype=np.int8)
    delta_current = M_current - m_actual
    
    # This expensive O(N^2) math happens EXACTLY ONCE
    t_shifted_current = t_bases + (C_matrices @ delta_current[..., None]).squeeze(-1)

    def compute_log_likelihood(M_prop, t_shifted):
        """Now takes t_shifted as an argument so we don't recalculate it"""
        delta = M_prop - m_actual
        t_isolated = t_shifted - (C_diag * delta)
        
        t_in = np.where(in_mask, t_isolated, np.nan)
        t_out = np.where(out_mask, t_isolated, np.nan)
        
        # Suppress RuntimeWarnings for all-NaN slices
        with np.errstate(invalid='ignore'):
            mu_in = np.nanmean(t_in, axis=0)   
            mu_out = np.nanmean(t_out, axis=0) 
            std_in = np.nanstd(t_in, axis=0, ddof=1)
            std_out = np.nanstd(t_out, axis=0, ddof=1)
        
        std_in = np.where(np.isnan(std_in) | (std_in == 0), global_std, std_in)
        std_out = np.where(np.isnan(std_out) | (std_out == 0), global_std, std_out)
        
        mu_target = np.where(M_prop == 1, mu_in, mu_out)
        std_target = np.where(M_prop == 1, std_in, std_out)
        
        log_liks = t.logpdf(target_scores, df=df, loc=mu_target, scale=std_target)
        return np.sum(log_liks / temperature)

    def compute_log_prior(M):
        return np.sum(M * log_p + (1 - M) * log_1_p)
    
    current_log_lik = compute_log_likelihood(M_current, t_shifted_current)
    current_log_prior = compute_log_prior(M_current)
    current_log_posterior = current_log_lik + current_log_prior
    
    # We use 'ab' (append binary) so we can write raw bytes continuously
    trace_file = open(trace_file_path, "wb")
    accepted_count = 0
    
    # --- 3. MCMC LOOP ---
    for step in range(num_steps):
        flip_idx = np.random.choice(valid_indices)
        
        # 1. Flip the bit
        M_prop = M_current.copy()
        M_prop[flip_idx] = 1 - M_prop[flip_idx]
        
        # 2. Incrementally update t_shifted in O(N) time!
        # If we flipped 0 -> 1, delta increased by 1. If 1 -> 0, delta decreased by 1.
        flip_direction = 1 if M_prop[flip_idx] == 1 else -1
        t_shifted_prop = t_shifted_current + (flip_direction * C_matrices[:, :, flip_idx])
        
        # 3. Compute new probabilities
        prop_log_lik = compute_log_likelihood(M_prop, t_shifted_prop)
        prop_log_prior = current_log_prior + flip_direction * (log_p - log_1_p) # O(1) prior update
        prop_log_posterior = prop_log_lik + prop_log_prior
        
        log_alpha = prop_log_posterior - current_log_posterior
        if np.isnan(log_alpha):
            log_alpha = -np.inf
            
        with np.errstate(over='ignore'):
            alpha = np.exp(np.float64(log_alpha))
            if np.isinf(alpha):
                alpha = 1.0
        alpha = min(1.0, alpha)
        
        # Accept/Reject
        if np.random.uniform(0, 1) < alpha:
            M_current = M_prop
            t_shifted_current = t_shifted_prop
            current_log_lik = prop_log_lik
            current_log_prior = prop_log_prior
            current_log_posterior = prop_log_posterior
            accepted_count += 1

        # Store sample in pre-allocated array
        
        trace_file.write(M_current.astype(np.int8).tobytes())
        
        if (step + 1) % 1000 == 0:
            print(f"  [MCMC] Step {step+1}/{num_steps} | Acceptance Rate: {accepted_count/(step+1):.2f}", flush=True)
            # Flush to disk every 1000 steps so it's safe if the job dies
            trace_file.flush()
            os.fsync(trace_file.fileno())

    trace_file.close()
    return trace_file_path



def aggregate_posterior(trace_path, num_points=2000, burn_in=200000):
    """
    Reads the binary MCMC trace from disk and computes the posterior probability 
    and standard deviation for each point.
    """
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"MCMC trace file not found: {trace_path}")

    # 1. Map the raw bytes from disk into a virtual 1D NumPy array (0 memory overhead)
    trace_map = np.memmap(trace_path, dtype=np.int8, mode='r')
    
    # 2. Automatically detect how many steps were actually written to the file
    # This prevents errors if the job was OOM-killed early!
    total_steps = len(trace_map) // num_points
    print(f"Aggregating {total_steps} recorded MCMC steps...")
    
    # 3. Reshape the 1D array into a 2D matrix: (Total Steps, Query Points)
    samples = trace_map.reshape((total_steps, num_points))
    
    # 4. Handle edge cases where burn-in is too long
    if burn_in >= total_steps:
        print("WARNING: Run crashed before burn-in finished! Using last 10% of samples.")
        burn_in = int(total_steps * 0.9)
        
    valid_samples = samples[burn_in:]
    
    # 5. Calculate probabilities and standard deviations 
    # Use dtype=np.float32 to save RAM during the aggregation math
    posterior_probs = np.mean(valid_samples, axis=0, dtype=np.float32)
    posterior_std = np.std(valid_samples, axis=0, dtype=np.float32)
    
    return posterior_probs, posterior_std


def save_attack_inputs(attack_dir, query_indices, ground_truth):
    """Save the query set and labels needed for offline evaluation."""
    np.savez(
        os.path.join(attack_dir, 'attack_data.npz'),
        query_indices=np.asarray(query_indices, dtype=np.int64),
        ground_truth=np.asarray(ground_truth, dtype=np.int8),
    )
    print(f"Saved attack inputs to {attack_dir}")


def load_attack_inputs(attack_dir):
    """Load query indices and ground-truth labels from a previous run."""
    attack_data_path = os.path.join(attack_dir, 'attack_data.npz')
    if not os.path.exists(attack_data_path):
        raise FileNotFoundError(
            f"Missing attack data file: {attack_data_path}. "
            "Cannot reuse this run for MCMC-only execution."
        )

    attack_data = np.load(attack_data_path)
    query_indices = attack_data['query_indices']
    ground_truth = attack_data['ground_truth']
    return query_indices, ground_truth

def setup_attack_directory(base_dir="attacks", model_name=None):
    """Creates or resumes an attack directory named after the target model."""
    attack_dir = os.path.join(base_dir, model_name)
    os.makedirs(attack_dir, exist_ok=True)
    return attack_dir

def load_target_metadata(meta_path):
    """Loads the exact indices the target model was trained on."""
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return set(meta['train_indices'])

def setup_query_points(meta_path, num_queries=1000, total_dataset_size=50000, member_percentage=0.5):
    """
    Randomly selects a set of query points to attack.
    Args:
        member_percentage: Percentage of query points that should be true members (default: 0.5 for 50%)
    Returns:
        query_indices: The absolute indices in the CIFAR-10 dataset to attack.
        ground_truth: Array of 1s (Member) and 0s (Non-Member) for evaluation.
    """
    # Figure out what the target model actually saw
    true_members = load_target_metadata(meta_path)
    all_indices = set(range(total_dataset_size))
    true_non_members = all_indices - true_members
    
    # Calculate number of members and non-members based on percentage
    num_members = int(num_queries * member_percentage)
    num_non_members = num_queries - num_members
    
    sampled_members = np.random.choice(list(true_members), num_members, replace=False)
    sampled_non_members = np.random.choice(list(true_non_members), num_non_members, replace=False)
    
    query_indices = np.concatenate([sampled_members, sampled_non_members])
    np.random.shuffle(query_indices)
    
    ground_truth = np.array([1 if idx in true_members else 0 for idx in query_indices])
    return query_indices, ground_truth



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MCMC-based Membership Inference Attack')
    parser.add_argument('--num_queries', type=int, default=1000,
                        help='Number of query points to attack (default: 1000)')
    parser.add_argument('--num_shadow_models', type=int, default=16,
                        help='Number of shadow models to train (default: 16)')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory containing the target model checkpoint and metadata')
    parser.add_argument('--prior-prob', type=float, required=True, help='Adversary\'s prior probability that any given point is a member, useful for preventing FP')
    parser.add_argument('--use-final-model', action='store_true',
                        help='Use final model instead of best model')
    parser.add_argument('--member-percentage', type=float, default=0.5,
                        help='Percentage of query points that are true members (default: 0.5 for 50%%)')
    parser.add_argument('--mcmc-steps', type=int, default=10000,
                        help='Number of MCMC steps (default: 10000)')
    parser.add_argument('--burn-in', type=int, default=2000,
                        help='Number of burn-in samples to discard when aggregating posterior (default: 2000)')
    parser.add_argument('--shadow-epochs', type=int, default=None,
                        help='Number of epochs to train shadow models (default: use target model epochs from metadata)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature scaling for likelihood computation (default: 0.7)')
    parser.add_argument('--attack-dir', type=str, default="attacks",
                        help='Base directory to store attack results (default: attacks)')
    parser.add_argument('--reuse-attack-run', type=str, default=None,
                        help='Reuse an existing attack run directory and skip shadow retraining/influence precompute')

    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reuse_mode = args.reuse_attack_run is not None
    checkpoint_dir = args.checkpoint_dir
    model_name = os.path.basename(checkpoint_dir.rstrip("/"))

    if reuse_mode:
        attack_dir = args.reuse_attack_run
        if not os.path.exists(attack_dir):
            raise FileNotFoundError(f"Reuse run directory not found: {attack_dir}")
    else:
        attack_dir = setup_attack_directory(base_dir=args.attack_dir, model_name=model_name)
    meta_path = os.path.join(checkpoint_dir, "training_metadata.json")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    model_suffix = "final" if args.use_final_model else "best"
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(f"_{model_suffix}.pth")]
    if not model_files:
        raise FileNotFoundError(f"No model file found with suffix '{model_suffix}' in {checkpoint_dir}")

    target_model_path = os.path.join(checkpoint_dir, model_files[0])

    print(f"Loading target model from: {target_model_path}")
    print(f"Loading metadata from: {meta_path}")

    NUM_SHADOW_MODELS = args.num_shadow_models
    MCMC_STEPS = args.mcmc_steps
    BURN_IN = args.burn_in
    PRIOR_PROB = args.prior_prob
    TEMPERATURE = args.temperature

    attack_data_path = os.path.join(attack_dir, 'attack_data.npz')
    if reuse_mode or os.path.exists(attack_data_path):
        query_indices, ground_truth = load_attack_inputs(attack_dir)
        NUM_QUERIES = len(query_indices)
        if not reuse_mode:
            print(f"Resuming existing run — loaded {NUM_QUERIES} query points from {attack_data_path}")
    else:
        NUM_QUERIES = args.num_queries
        query_indices, ground_truth = setup_query_points(
            meta_path,
            num_queries=NUM_QUERIES,
            member_percentage=args.member_percentage,
        )
        save_attack_inputs(attack_dir, query_indices, ground_truth)

    print(f"Attack configuration:")
    print(f"  - Checkpoint Directory: {checkpoint_dir}")
    print(f"  - Target Model: {target_model_path}")
    print(f"  - Attack Directory: {attack_dir}")
    print(f"  - Reuse Existing Run: {'yes' if reuse_mode else 'no'}")
    print(f"  - Num Queries: {NUM_QUERIES}")
    print(f"  - Num Shadow Models: {NUM_SHADOW_MODELS}")
    print(f"  - Prior Membership Probability: {PRIOR_PROB:.4f}")
    print(f"  - MCMC Steps: {MCMC_STEPS}")
    print(f"  - Burn-in: {BURN_IN}")
    print(f"  - Temperature: {TEMPERATURE}")
    print(f"  - Target FPRs: 0.1%, 1.0%")
    if not reuse_mode:
        print(f"  - Member Percentage: {args.member_percentage * 100:.1f}%")
    
    # Load target metadata for dataset sizes and training configuration
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    TOTAL_DATASET_SIZE = meta['total_cifar10_train_size']
    TARGET_TRAIN_SIZE = meta['num_samples_used']
    
    training_config = None
    if not reuse_mode:
        # Load training configuration from metadata
        print(f"\nLoading training configuration from metadata...")
        training_config = TrainingConfig.from_metadata(meta_path)

        # Override epochs if specified
        if args.shadow_epochs is not None:
            print(f"  Overriding epochs: {training_config.epochs} -> {args.shadow_epochs}")
            training_config.epochs = args.shadow_epochs

        print(f"  Shadow models will use:")
        print(f"    - Optimizer: {training_config.optimizer_type.upper()}")
        print(f"    - Learning rate: {training_config.lr}")
        print(f"    - Epochs: {training_config.epochs}")
        print(f"    - Batch size: {training_config.batch_size}")
        print(f"    - Scheduler: {training_config.scheduler_type}")
    
    print(f"\nTarget model was trained on {TARGET_TRAIN_SIZE} samples out of {TOTAL_DATASET_SIZE}.")
    print(f"Starting attack on {len(query_indices)} points.")
    print(f"Number of true members in query set: {np.sum(ground_truth)}")

    shadow_models_dir = os.path.join(attack_dir, "shadow_models")
    precomputed_dir = os.path.join(attack_dir, "precomputed_matrices")
    os.makedirs(shadow_models_dir, exist_ok=True)
    os.makedirs(precomputed_dir, exist_ok=True)

    influence_cache_path = os.path.join(precomputed_dir, "influence_data.npz")
    m_actual_path = os.path.join(shadow_models_dir, "m_actual.npy")

    if reuse_mode:
        if not os.path.exists(influence_cache_path):
            raise FileNotFoundError(
                f"Missing influence cache at {influence_cache_path}. "
                "Run without --reuse-attack-run once to build shadow models/influence matrices first."
            )
        if not os.path.exists(m_actual_path):
            raise FileNotFoundError(
                f"Missing membership matrix at {m_actual_path}. "
                "Run without --reuse-attack-run once to build shadow models first."
            )

        print(f"Reusing cached shadow artifacts from {attack_dir}...")
        influence_data = np.load(influence_cache_path)
        C_matrices = influence_data['C_matrices']
        t_bases = influence_data['t_bases']
        m_actual = np.load(m_actual_path)

        if m_actual.shape[1] != len(query_indices):
            raise ValueError(
                f"Mismatch between cached memberships ({m_actual.shape[1]} points) and "
                f"attack_data.npz ({len(query_indices)} points)."
            )
    else:
        shadow_models, m_actual, shadow_subsets, any_new_models_trained = train_shadow_models(
            query_indices=query_indices,
            num_models=NUM_SHADOW_MODELS,
            total_dataset_size=TOTAL_DATASET_SIZE,
            target_train_size=TARGET_TRAIN_SIZE,
            training_config=training_config,
            confident_memberships=None,
            shadow_models_dir=shadow_models_dir
        )

        np.save(m_actual_path, m_actual)

        for k, model in enumerate(shadow_models):
            torch.save({
                'model_state_dict': model.state_dict(),
                'm_k': m_actual[k],
                'subset_indices': shadow_subsets[k]
            }, os.path.join(shadow_models_dir, f"shadow_{k}.pth"))

        if (not any_new_models_trained) and os.path.exists(influence_cache_path):
            print(f"Done loading shadow models. Reusing cached influence data from {influence_cache_path}...")
            influence_data = np.load(influence_cache_path)
            C_matrices = influence_data['C_matrices']
            t_bases = influence_data['t_bases']
        else:
            print("Done training/loading shadow models. Now precomputing influence matrices...")
            C_matrices, t_bases = precompute_influence_matrices(
                shadow_models=shadow_models,
                shadow_subsets=shadow_subsets,
                query_indices=query_indices,
                device=device,
                hessian_sample_size=5000
            )

            np.savez_compressed(
                influence_cache_path,
                C_matrices=C_matrices,
                t_bases=t_bases
            )

        del shadow_models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Evaluating target model on query points...")
    target_scores = evaluate_target_model(target_model_path, query_indices, device)

    print(f"Running MCMC for {NUM_QUERIES} points...")
    trace_path = run_mcmc(
        target_scores=target_scores,
        C_matrices=C_matrices,
        t_bases=t_bases,
        m_actual=m_actual,
        output_dir=attack_dir,
        num_steps=MCMC_STEPS,
        prior_prob=PRIOR_PROB,
        temperature=TEMPERATURE
    )

    posterior_probs, posterior_std = aggregate_posterior(
        trace_path,
        num_points=len(target_scores),
        burn_in=BURN_IN
    )

    posterior_dict = {str(q_idx): float(prob) for q_idx, prob in zip(query_indices, posterior_probs)}
    with open(os.path.join(attack_dir, "posterior_probs.json"), "w") as f:
        json.dump(posterior_dict, f, indent=2)

    np.save(os.path.join(attack_dir, "posterior_std.npy"), posterior_std)

    fpr_curve, tpr_curve, _ = roc_curve(ground_truth, posterior_probs)
    valid_01 = np.where(fpr_curve <= 0.001)[0]
    mcmc_tpr_01pct = tpr_curve[valid_01[-1]] if len(valid_01) > 0 else float('nan')
    valid_1 = np.where(fpr_curve <= 0.01)[0]
    mcmc_tpr_1pct = tpr_curve[valid_1[-1]] if len(valid_1) > 0 else float('nan')

    _, lira_tpr_01pct, lira_tpr_1pct = run_lira_baseline(
        target_scores,
        t_bases,
        m_actual,
        ground_truth,
        attack_dir,
    )

    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"  MCMC-CMIA:      TPR @ 0.1% FPR = {mcmc_tpr_01pct * 100:.2f}%  |  TPR @ 1% FPR = {mcmc_tpr_1pct * 100:.2f}%")
    print(f"  LiRA (control): TPR @ 0.1% FPR = {lira_tpr_01pct * 100:.2f}%  |  TPR @ 1% FPR = {lira_tpr_1pct * 100:.2f}%")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
