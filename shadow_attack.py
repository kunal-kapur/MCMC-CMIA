import os
import json
import gc
import shutil
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
        Tuple of (shadow_models, shadow_datasets_m, shadow_subsets)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shadow_models = []
    shadow_datasets_m = [] 
    shadow_subsets = []
    
    all_indices = set(range(total_dataset_size))
    background_pool = list(all_indices - set(query_indices))
    
    # Check if we're resuming from a previous run
    membership_matrix_path = None
    if shadow_models_dir is not None:
        os.makedirs(shadow_models_dir, exist_ok=True)
        membership_matrix_path = os.path.join(shadow_models_dir, 'membership_matrix.npy')
    
    # Load or create membership matrix
    if membership_matrix_path and os.path.exists(membership_matrix_path):
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
        if shadow_models_dir is not None:
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
            
    return shadow_models, np.array(shadow_datasets_m), shadow_subsets


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

def run_mcmc_block(target_scores, C_matrices, t_bases, m_actual, anchors, block_dir, num_steps=10000, prior_prob=0.5, temperature=0.7):
    N = len(target_scores)
    K = len(C_matrices)
    global_std = np.std(t_bases, ddof=1) + 1e-8
    
    # --- 1. PRECOMPUTE STATIC VARIABLES ONCE ---
    log_p = np.log(prior_prob + 1e-10)
    log_1_p = np.log(1 - prior_prob + 1e-10)
    
    valid_indices = np.where(anchors == -1)[0]
    trace_file_path = os.path.join(block_dir, "mcmc_trace.npy")
    
    if len(valid_indices) == 0:
        # If all anchors are known, write static array to disk and return path
        static_trace = np.tile(anchors.astype(np.int8), (num_steps, 1))
        with open(trace_file_path, "wb") as f:
            f.write(static_trace.tobytes())
        return trace_file_path
        
    C_diag = np.diagonal(C_matrices, axis1=1, axis2=2)
    in_mask = (m_actual == 1)
    out_mask = (m_actual == 0)
    df = (K / 2) - 1
    
    # --- 2. INITIALIZE STATE ---
    M_current = np.random.randint(0, 2, size=N, dtype=np.int8)
    for i in range(N):
        if anchors[i] != -1:
            M_current[i] = anchors[i]
            
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


def get_new_anchors(posterior_probs, posterior_std, current_anchors, 
                    threshold_in=0.999, threshold_out=0.001, max_std=0.05):
    """
    Identify points we are highly confident about, filtering out unstable MCMC predictions.
    
    Args:
        posterior_probs: Mean posterior probability for each point
        posterior_std: Standard deviation of the MCMC samples for each point
        current_anchors: Existing anchors array
        threshold_in: Minimum probability to anchor as a member
        threshold_out: Maximum probability to anchor as a non-member
        max_std: Maximum allowed standard deviation to consider the prediction stable
    """
    new_anchors = current_anchors.copy()
    
    for i in range(len(posterior_probs)):
        # Skip points that are already anchored
        if current_anchors[i] != -1:
            continue
            
        prob = posterior_probs[i]
        std = posterior_std[i]
        
        # Only anchor if the MCMC chain was stable
        if std <= max_std:
            if prob >= threshold_in:
                new_anchors[i] = 1
            elif prob <= threshold_out:
                new_anchors[i] = 0
                
    return new_anchors

def save_attack_metadata(attack_dir, args, query_indices, ground_truth, anchors, target_model_path, meta_path):
    """
    Saves configuration and query setup for potential resume.
    """
    config = {
        'num_queries': args.num_queries,
        'num_shadow_models': args.num_shadow_models,
        'num_blocks': args.num_blocks,
        'checkpoint_dir': args.checkpoint_dir,
        'use_final_model': args.use_final_model,
        'member_percentage': args.member_percentage,
        'mcmc_steps': args.mcmc_steps,
        'burn_in': args.burn_in,
        'confidence_threshold': args.confidence_threshold,
        'temperature': args.temperature,
        'target_model_path': target_model_path,
        'meta_path': meta_path,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(attack_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    np.save(os.path.join(attack_dir, 'query_indices.npy'), query_indices)
    np.save(os.path.join(attack_dir, 'ground_truth.npy'), ground_truth)
    np.save(os.path.join(attack_dir, 'initial_anchors.npy'), anchors)
    
    print(f"Saved attack metadata to {attack_dir}")

def load_checkpoint_state(resume_dir):
    """
    Loads state from a previous attack run to resume from.
    Returns: (config, query_indices, ground_truth, anchors, last_completed_block)
    """
    print(f"Loading checkpoint from {resume_dir}...")
    
    # Load config
    with open(os.path.join(resume_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Load query setup
    query_indices = np.load(os.path.join(resume_dir, 'query_indices.npy'))
    ground_truth = np.load(os.path.join(resume_dir, 'ground_truth.npy'))
    
    # Find the last completed block by checking which block directories exist
    last_completed_block = -1
    for block_num in range(config['num_blocks']):
        block_name = f"block_{block_num}_random_init" if block_num == 0 else f"block_{block_num}_cascade"
        block_dir = os.path.join(resume_dir, block_name)
        
        # Check if block was fully completed (has anchors_for_next_block.npy)
        if os.path.exists(os.path.join(block_dir, 'anchors_for_next_block.npy')):
            last_completed_block = block_num
        else:
            break
    
    # Load anchors from the last completed block, or initial if none completed
    if last_completed_block >= 0:
        last_block_name = f"block_{last_completed_block}_random_init" if last_completed_block == 0 else f"block_{last_completed_block}_cascade"
        last_block_dir = os.path.join(resume_dir, last_block_name)
        anchors = np.load(os.path.join(last_block_dir, 'anchors_for_next_block.npy'))
        print(f"Resuming from block {last_completed_block + 1} (last completed: block {last_completed_block})")
    else:
        anchors = np.load(os.path.join(resume_dir, 'initial_anchors.npy'))
        print(f"No completed blocks found, starting from block 0")
    
    return config, query_indices, ground_truth, anchors, last_completed_block + 1

def setup_attack_directory(base_dir="attacks"):
    """Creates a unique timestamped directory for this attack run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # TODO ADJUST, KEEPING SO I DONT RETRAIN MODEL
    timestamp = "20260227_111315" # TEMPORARY OVERRIDE FOR TESTING
    attack_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(attack_dir, exist_ok=True)
    return attack_dir

def setup_block_directory(attack_dir, block_num):
    """Creates subfolders for a specific MCMC block."""
    block_name = f"block_{block_num}_random_init" if block_num == 0 else f"block_{block_num}_cascade"
    block_dir = os.path.join(attack_dir, block_name)
    
    os.makedirs(block_dir, exist_ok=True)
    os.makedirs(os.path.join(block_dir, "shadow_models"), exist_ok=True)
    os.makedirs(os.path.join(block_dir, "precomputed_matrices"), exist_ok=True)
    
    return block_dir

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
        anchors: Array initialized to -1 (uncertain) to track MCMC confidence.
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
    
    anchors = np.full(num_queries, -1) # 1 for member, 0 for non-member, -1 for uncertain (used in cascade)
    
    return query_indices, ground_truth, anchors

def load_existing_shadow_models(shadow_models_dir, num_models):
    shadow_models = []
    shadow_subsets = []
    
    # Load the global m_actual matrix
    m_actual_path = os.path.join(shadow_models_dir, "m_actual.npy")
    m_actual = np.load(m_actual_path)
    
    for k in range(num_models):
        model_path = os.path.join(shadow_models_dir, f"shadow_{k}.pth")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Initialize a new model architecture
        model = ResNet18_Influence()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        shadow_models.append(model)
        
        # Extract the saved training subset indices
        # Fallback to an empty list or error if trying to load an old checkpoint
        if 'subset_indices' in checkpoint:
            shadow_subsets.append(checkpoint['subset_indices'])
        else:
            raise KeyError(f"Checkpoint {model_path} does not contain 'subset_indices'. You must retrain the models.")
            
    return shadow_models, m_actual, shadow_subsets



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MCMC-based Membership Inference Attack')
    parser.add_argument('--num_queries', type=int, default=1000,
                        help='Number of query points to attack (default: 1000)')
    parser.add_argument('--num_shadow_models', type=int, default=16,
                        help='Number of shadow models to train per block (default: 16)')
    parser.add_argument('--num_blocks', type=int, default=3,
                        help='Number of MCMC blocks to run (default: 3)')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory containing the target model checkpoint and metadata')
    parser.add_argument('--prior-prob', type=float, required=True, help='Adversary\'s prior probability that any given point is a member, useful for preventing FP')
    parser.add_argument('--use-final-model', action='store_true',
                        help='Use final model instead of best model')
    parser.add_argument('--member-percentage', type=float, default=0.5,
                        help='Percentage of query points that are true members (default: 0.5 for 50%%)')
    parser.add_argument('--mcmc-steps', type=int, default=10000,
                        help='Number of MCMC steps per block (default: 10000)')
    parser.add_argument('--burn-in', type=int, default=2000,
                        help='Number of burn-in samples to discard when aggregating posterior (default: 2000)')
    parser.add_argument('--confidence-threshold', type=float, default=0.999,
                        help='Confidence threshold for anchoring points (default: 0.999 for 99.9%%)')
    parser.add_argument('--shadow-epochs', type=int, default=None,
                        help='Number of epochs to train shadow models (default: use target model epochs from metadata)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume attack from a previous run directory (overrides other arguments)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature scaling for likelihood computation (default: 0.7)')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if resuming from checkpoint
    if args.resume_from:
        print("=" * 60)
        print("RESUMING FROM CHECKPOINT")
        print("=" * 60)
        
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"Resume directory not found: {args.resume_from}")
        
        # Load checkpoint state
        config, query_indices, ground_truth, anchors, start_block = load_checkpoint_state(args.resume_from)
        
        # Use the same attack directory (resume in place)
        attack_dir = args.resume_from
        
        # Extract configuration from loaded config
        checkpoint_dir = config['checkpoint_dir']
        target_model_path = config['target_model_path']
        meta_path = config['meta_path']
        NUM_QUERIES = config['num_queries']
        NUM_SHADOW_MODELS = config['num_shadow_models']
        NUM_BLOCKS = config['num_blocks']
        MCMC_STEPS = config['mcmc_steps']
        BURN_IN = config['burn_in']
        CONFIDENCE_THRESHOLD = config.get('confidence_threshold', 0.999)
        PRIOR_PROB = config['prior_prob']
        TEMPERATURE = config.get('temperature', 0.7)
        
        print(f"Loaded configuration from checkpoint:")
        print(f"  - Attack Directory: {attack_dir}")
        print(f"  - Num Queries: {NUM_QUERIES}")
        print(f"  - Num Shadow Models: {NUM_SHADOW_MODELS}")
        print(f"  - Num Blocks: {NUM_BLOCKS}")
        print(f"  - MCMC Steps: {MCMC_STEPS}")
        print(f"  - Burn-in: {BURN_IN}")
        print(f"  - Confidence Threshold: {CONFIDENCE_THRESHOLD}")
        print(f"  - Temperature: {TEMPERATURE}")
        print(f"  - Starting from Block: {start_block}")
        print(f"  - Anchored Points: {np.sum(anchors != -1)} / {len(anchors)}")
        print(f"  - Prior Membership Probability: {PRIOR_PROB:.4f}")
        
    else:
        # Fresh start - normal initialization
        attack_dir = setup_attack_directory()
        
        # Auto-discover metadata and model files from checkpoint directory
        checkpoint_dir = args.checkpoint_dir
        meta_path = os.path.join(checkpoint_dir, "training_metadata.json")
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        # Determine which model to use
        model_suffix = "final" if args.use_final_model else "best"
        
        # Find model file in checkpoint directory
        model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(f"_{model_suffix}.pth")]
        if not model_files:
            raise FileNotFoundError(f"No model file found with suffix '{model_suffix}' in {checkpoint_dir}")
        
        target_model_path = os.path.join(checkpoint_dir, model_files[0])
        
        print(f"Loading target model from: {target_model_path}")
        print(f"Loading metadata from: {meta_path}")
        
        NUM_QUERIES = args.num_queries
        NUM_SHADOW_MODELS = args.num_shadow_models
        NUM_BLOCKS = args.num_blocks
        MCMC_STEPS = args.mcmc_steps
        BURN_IN = args.burn_in
        CONFIDENCE_THRESHOLD = args.confidence_threshold
        PRIOR_PROB = args.prior_prob
        TEMPERATURE = args.temperature
        print(f"Attack configuration:")
        print(f"  - Checkpoint Directory: {checkpoint_dir}")
        print(f"  - Target Model: {target_model_path}")
        print(f"  - Num Queries: {NUM_QUERIES}")
        print(f"  - Num Shadow Models: {NUM_SHADOW_MODELS}")
        print(f"  - Prior Membership Probability: {PRIOR_PROB:.4f}")
        print(f"  - Num Blocks: {NUM_BLOCKS}")
        print(f"  - Confidence Threshold: {CONFIDENCE_THRESHOLD}")
        print(f"  - MCMC Steps: {MCMC_STEPS}")
        print(f"  - Burn-in: {BURN_IN}")
        print(f"  - Temperature: {TEMPERATURE}")
        print(f"  - Member Percentage: {args.member_percentage * 100:.1f}%")
        
        query_indices, ground_truth, anchors = setup_query_points(meta_path, num_queries=NUM_QUERIES, member_percentage=args.member_percentage)
        
        # Save metadata for potential resume
        save_attack_metadata(attack_dir, args, query_indices, ground_truth, anchors, target_model_path, meta_path)
        
        start_block = 0
    
    # Load target metadata for dataset sizes and training configuration
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    TOTAL_DATASET_SIZE = meta['total_cifar10_train_size']
    TARGET_TRAIN_SIZE = meta['num_samples_used']
    
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
    
    for block in range(start_block, NUM_BLOCKS):
        print(f"\n{'='*40}")
        print(f"--- Starting Block {block} ---")
        print(f"Current Anchors: {np.sum(anchors != -1)} / {len(anchors)}")
        print(f"{'='*40}")
        
        block_dir = setup_block_directory(attack_dir, block)
        
        # Shadow models directory for this block
        shadow_models_dir = os.path.join(block_dir, "shadow_models")
        os.makedirs(shadow_models_dir, exist_ok=True)
        
        # Train shadow models (with automatic resumption if models already exist)
        shadow_models, m_actual, shadow_subsets = train_shadow_models(
            query_indices=query_indices,
            num_models=NUM_SHADOW_MODELS,
            total_dataset_size=TOTAL_DATASET_SIZE,
            target_train_size=TARGET_TRAIN_SIZE,
            training_config=training_config,
            confident_memberships=anchors,
            shadow_models_dir=shadow_models_dir
        )
        
        # Save m_actual matrix globally
        np.save(os.path.join(shadow_models_dir, "m_actual.npy"), m_actual)
        
        # Save models AND their exact training subsets
        for k, model in enumerate(shadow_models):
            torch.save({
                'model_state_dict': model.state_dict(),
                'm_k': m_actual[k],
                'subset_indices': shadow_subsets[k]
            }, os.path.join(block_dir, "shadow_models", f"shadow_{k}.pth"))

        print("Done training shadow models. Now precomputing influence matrices...")
        C_matrices, t_bases = precompute_influence_matrices(
            shadow_models=shadow_models, 
            shadow_subsets=shadow_subsets,
            query_indices=query_indices, 
            device=device,
            hessian_sample_size=5000
        )
        
        np.savez_compressed(
            os.path.join(block_dir, "precomputed_matrices", "influence_data.npz"), 
            C_matrices=C_matrices, 
            t_bases=t_bases
        )

        print("Evaluating target model on query points...")
        
        target_scores = evaluate_target_model(target_model_path, query_indices, device)
        
        print(f"Running MCMC for {NUM_QUERIES} points...")

        # Assume adversary has good prior on query
        trace_path = run_mcmc_block(
            target_scores=target_scores,
            C_matrices=C_matrices, 
            t_bases=t_bases, 
            m_actual=m_actual, 
            anchors=anchors, 
            block_dir=block_dir,
            num_steps=MCMC_STEPS,
            prior_prob=PRIOR_PROB,  # Use the prior probability passed in via args
            temperature=TEMPERATURE
        )

        # Aggregate step
        posterior_probs, posterior_std = aggregate_posterior(
            trace_path, 
            num_points=len(target_scores), 
            burn_in=200000
        )

        posterior_dict = {str(q_idx): float(prob) for q_idx, prob in zip(query_indices, posterior_probs)}
        with open(os.path.join(block_dir, "posterior_probs.json"), "w") as f:
            json.dump(posterior_dict, f, indent=2)
        
        # Print confidence table
        print(f"\n{'='*80}")
        print(f"BLOCK {block} RESULTS: Membership Predictions vs Ground Truth")
        print(f"{'='*80}")
        
        # Determine predictions based on posterior probabilities
        predictions = []
        for prob in posterior_probs:
            if prob >= 0.99:
                predictions.append('1')  # Confident: IN
            elif prob <= 0.01:
                predictions.append('0')  # Confident: OUT
            else:
                predictions.append('?')  # Uncertain
        
        # Print table header (first 20 points)
        num_to_show = min(100, len(query_indices))
        header = "Point:      " + "  ".join([f"{i+1:>3}" for i in range(num_to_show)])
        print(header)
        print("-" * len(header))
        pred_row = "Predicted:  " + "  ".join([f"{p:>3}" for p in predictions[:num_to_show]])
        print(pred_row)
        
        truth_row = "Actual:     " + "  ".join([f"{int(gt):>3}" for gt in ground_truth[:num_to_show]])
        print(truth_row)

        posterior_row = "Posterior:  " + "  ".join([f"{prob:.2f}" for prob in posterior_probs[:num_to_show]])
        print(posterior_row)
        
        # confusion matrix
        tp = sum(1 for p, gt in zip(predictions, ground_truth) if p == '1' and gt == 1)
        fp = sum(1 for p, gt in zip(predictions, ground_truth) if p == '1' and gt == 0)
        tn = sum(1 for p, gt in zip(predictions, ground_truth) if p == '0' and gt == 0)
        fn = sum(1 for p, gt in zip(predictions, ground_truth) if p == '0' and gt == 1)
        
        correct = tp + tn
        confident_predictions = sum(1 for p in predictions if p != '?')
        accuracy = correct / confident_predictions if confident_predictions > 0 else 0
        
        # Calculate TPR and FPR
        actual_positives = tp + fn
        actual_negatives = fp + tn
        tpr = tp / actual_positives if actual_positives > 0 else 0
        fpr = fp / actual_negatives if actual_negatives > 0 else 0
        
        print(f"\nConfident predictions: {confident_predictions}/{len(query_indices)}")
        print(f"Accuracy on confident predictions: {accuracy:.2%} ({correct}/{confident_predictions})")
        print(f"True Positive Rate (TPR): {tpr:.2%} ({tp}/{actual_positives})")
        print(f"False Positive Rate (FPR): {fpr:.2%} ({fp}/{actual_negatives})")
        print(f"{'='*80}\n")

        fpr, tpr, thresholds = roc_curve(ground_truth, posterior_probs)

        # Find the TPR where FPR is strictly <= 0.001 (0.1%)
        target_fpr = 0.001
        valid_indices = np.where(fpr <= target_fpr)[0]

        if len(valid_indices) > 0:
            tpr_at_low_fpr = tpr[valid_indices[-1]]
            print(f"Success! TPR at 0.1% FPR: {tpr_at_low_fpr * 100:.2f}%")
        else:
            print("Could not measure 0.1% FPR. Need more non-member points or shadow models.")
                    
        # Cascade (new anchors for next block)
        anchors = get_new_anchors(
            posterior_probs=posterior_probs, 
            posterior_std=posterior_std, 
            current_anchors=anchors, 
            threshold_in=CONFIDENCE_THRESHOLD, 
            threshold_out=(1 - CONFIDENCE_THRESHOLD),
            max_std=0.05  # Strict variance limit for anchoring
        )
        np.save(os.path.join(block_dir, 'anchors_for_next_block.npy'), anchors)
        #TODO REMOVE TO CHECK FIRST ITERATION
        exit(0)
        
        np.save(os.path.join(block_dir, "anchors_for_next_block.npy"), anchors)

if __name__ == "__main__":
    main()
