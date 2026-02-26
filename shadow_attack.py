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

import torchvision
from global_variables import DATA_DIR, TRANSFORM_TRAIN, TRANSFORM_TEST
import torchvision.datasets as datasets

from resnet import ResNet18
from resnet_influence import ResNet18_Influence
from train_utils import TrainingConfig, train_shadow_model

def evaluate_target_model(model_path, query_indices, device):
    print(f"Loading Target Model from {model_path}...")
    
    model = ResNet18()
    
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
            batch_size = logits.size(0)
            
            # Get the logit of the true class
            true_logits = logits[torch.arange(batch_size), targets]

            # Mask true class by setting it to negative infinity
            logits_clone = logits.clone()
            logits_clone[torch.arange(batch_size), targets] = -float('inf')

            # Compute the exact LiRA score
            scores = (true_logits - torch.logsumexp(logits_clone, dim=1)).cpu().numpy()
            target_scores.extend(scores)
            
    return np.array(target_scores)



def train_single_shadow_model(subset_indices, device, training_config):
    """Physically trains ONE ResNet on the specified subset of CIFAR-10.
    
    Args:
        subset_indices: List of training sample indices
        device: torch device
        training_config: TrainingConfig object from target model metadata
    
    Returns:
        Trained model in eval mode
    """
    model = ResNet18_Influence().to(device)
    trained_model = train_shadow_model(model, subset_indices, training_config, device)
    return trained_model

def train_shadow_models(query_indices, num_models, total_dataset_size, target_train_size, 
                       training_config, confident_memberships=None):
    """
    Train shadow models using the same configuration as the target model.
    
    Args:
        query_indices: Indices of query points
        num_models: Number of shadow models to train
        total_dataset_size: Total size of CIFAR-10 training set
        target_train_size: Size of target model's training set
        training_config: TrainingConfig object from target model metadata
        confident_memberships: Optional array of anchored memberships (-1 for unknown, 0/1 for known)
    
    Returns:
        Tuple of (shadow_models, shadow_datasets_m)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shadow_models = []
    shadow_datasets_m = [] 
    
    all_indices = set(range(total_dataset_size))
    background_pool = list(all_indices - set(query_indices))
    
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
    
    # Now train each shadow model based on the pre-computed membership matrix
    for k in range(num_models):
        print(f"  -> Training Shadow Model {k+1}/{num_models}...")
        
        m_k = membership_matrix[k]
        subset_indices = []
        
        # Add query points that are IN for this model
        for i, idx in enumerate(query_indices):
            if m_k[i] == 1:
                subset_indices.append(int(idx))
                
        num_background_needed = target_train_size - len(subset_indices)
        
        # Randomly sample from the background pool
        background_sample = np.random.choice(background_pool, num_background_needed, replace=False)
        subset_indices.extend(background_sample.tolist())
        
        
        trained_model = train_single_shadow_model(subset_indices, device, training_config)
        
        shadow_models.append(trained_model.cpu()) # this makes copy of model in CPU
        shadow_datasets_m.append(m_k)
        
        del trained_model 
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return shadow_models, np.array(shadow_datasets_m)


def precompute_influence_matrices(shadow_models, m_actual_matrix, query_indices, device):
    """
    Calculate Hessian and Influence matrix C for all shadow models.
    """
    C_matrices = []
    t_bases = []
    
    full_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=TRANSFORM_TEST)
    
    print("Starting")
    for k, model in enumerate(shadow_models):
        print(f"  -> Precomputing Influence for Shadow Model {k+1}/{len(shadow_models)}...")
        
        # Reconstruct the exact dataset this shadow model was trained on
        m_k = m_actual_matrix[k]
        subset_indices = []
        for i, is_in in enumerate(m_k):
            if is_in == 1:
                subset_indices.append(int(query_indices[i]))
        
        subset = Subset(full_dataset, subset_indices)
        
        loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=0)
        
        model = model.to(device)
        
        # Compute and invert Hessian
        print("Flag 1")
        H = model.compute_last_layer_hessian(loader, device)
        H_inv = torch.linalg.inv(H)
        
        # Compute C matrix (the fast-forward Taylor approximation)
        # Note: You compute C using the QUERY points loader, not the training subset loader!
        print("Flag 2")
        query_subset = Subset(full_dataset, [int(idx) for idx in query_indices])
        query_loader = DataLoader(query_subset, batch_size=256, shuffle=False)
        print("Flag 3")
        
        C = model.compute_influence_matrix(query_loader, H_inv, device)
        C_matrices.append(C.cpu().numpy())
        
        t_base = model.get_lira_statistics(query_loader, device) 
        t_bases.append(t_base.cpu().numpy())
        print("Flag 4")
        
        # memory clean up
        model = model.cpu()
        del H, H_inv, C
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Done")
    print("HERE")
    return np.array(C_matrices), np.array(t_bases)


def run_mcmc_block(target_scores, C_matrices, t_bases, m_actual, anchors, num_steps=10000, prior_prob=0.5):
    """
    Runs Metropolis-Hastings to sample the joint membership posterior.
    
    Args:
        target_scores: (N,) array of LiRA scores from the target model
        C_matrices: (K, N, N) stacked precomputed influence matrices
        t_bases: (K, N) stacked base LiRA scores from the shadow models
        m_actual: (K, N) boolean array of what the shadow models actually saw
        anchors: (N,) array of known memberships (1=IN, 0=OUT, -1=Uncertain)
        prior_prob: Prior probability that any point is in the training set (default: 0.5)
    """
    N = len(target_scores)
    K = len(C_matrices)
    global_std = np.std(t_bases, ddof=1) + 1e-8
    
    # Precompute log prior terms for efficiency
    # Prior: each M_i ~ Bernoulli(prior_prob), IID
    # log P(M) = Σ [M_i * log(p) + (1-M_i) * log(1-p)]
    log_p = np.log(prior_prob + 1e-10)  # avoid log(0)
    log_1_p = np.log(1 - prior_prob + 1e-10)
    
    # start with random guess for start of MCMC
    M_current = np.random.randint(0, 2, size=N)
    for i in range(N):
        if anchors[i] != -1:
            M_current[i] = anchors[i]
            
    # Helper function for log-likelihood
    def compute_log_likelihood(M_prop):
        delta = M_prop - m_actual
        
        t_shifted = t_bases + (C_matrices @ delta[..., None]).squeeze(-1)
        
        total_log_lik = 0.0
        
        for i in range(N):
            # Can't shift models based on point i's proposed membership
            # revert the shift for point i to isolate the background effect
            t_shifted_i = t_shifted[:, i] - (C_matrices[:, i, i] * delta[:, i])
            
            # Split the K models based on whether they actually saw point i
            in_mask = (m_actual[:, i] == 1)
            out_mask = ~in_mask
            
            # Extract scores
            scores_in = t_shifted_i[in_mask]
            scores_out = t_shifted_i[out_mask]

            # TODO FIX IF NOT MODEL QUERY
            if M_prop[i] == 1:
                mu_in = np.mean(scores_in)
                # FIX: Use the stable global_std instead of the noisy local std
                log_lik = norm.logpdf(target_scores[i], loc=mu_in, scale=global_std)
            else:
                mu_out = np.mean(scores_out)
                # FIX: Use the stable global_std instead of the noisy local std
                log_lik = norm.logpdf(target_scores[i], loc=mu_out, scale=global_std)
                
            total_log_lik += log_lik
            
        return total_log_lik
    
    def compute_log_prior(M):
        """Compute log prior for membership vector M under IID Bernoulli(prior_prob) assumption."""
        return np.sum(M * log_p + (1 - M) * log_1_p)
    
    current_log_lik = compute_log_likelihood(M_current)
    current_log_prior = compute_log_prior(M_current)
    current_log_posterior = current_log_lik + current_log_prior
    
    samples = []
    accepted_count = 0
    
    # Metropolis Hastings Loop
    for step in range(num_steps):

        valid_indices = np.where(anchors == -1)[0]
        if len(valid_indices) == 0:
            samples.append(M_current.copy())
            continue
            
        # TODO investigate if better to just arbitrarily flip (feel like that is less stable)
        # random proposal to bit flip one at a time for now
        flip_idx = np.random.choice(valid_indices)
        
        M_prop = M_current.copy()
        M_prop[flip_idx] = 1 - M_prop[flip_idx]
        
        prop_log_lik = compute_log_likelihood(M_prop)
        prop_log_prior = compute_log_prior(M_prop)
        prop_log_posterior = prop_log_lik + prop_log_prior
        
        # Metropolis-Hastings Acceptance Ratio (log posterior ratio)
        log_alpha = prop_log_posterior - current_log_posterior
        
        # Accept if with prob exp(log_alpha) if worse
        if np.log(np.random.uniform(0, 1)) < log_alpha:
            M_current = M_prop
            current_log_lik = prop_log_lik
            current_log_prior = prop_log_prior
            current_log_posterior = prop_log_posterior
            accepted_count += 1
            
        samples.append(M_current.copy())
        
        if (step + 1) % 1000 == 0:
            print(f"  [MCMC] Step {step+1}/{num_steps} | Acceptance Rate: {accepted_count/(step+1):.2f}")
            
    return np.array(samples)


def aggregate_posterior(samples, burn_in=2000):
    """
    Calculate the marginal posterior probabilities.
    """
    valid_samples = samples[burn_in:]
    posterior_probs = np.mean(valid_samples, axis=0)
    return posterior_probs

def get_new_anchors(posterior_probs, current_anchors, threshold_in=0.999, threshold_out=0.001):
    """
    Identify points we are highly confident about.
    """
    new_anchors = current_anchors.copy()
    for i, prob in enumerate(posterior_probs):
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
    timestamp = "20260225_152614"
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
    m_actual = []
    
    model_files = sorted([f for f in os.listdir(shadow_models_dir) if f.endswith('.pth')])
    
    for model_file in model_files[:num_models]:
        model_path = os.path.join(shadow_models_dir, model_file)
        print(f"    Loading shadow model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        model = ResNet18_Influence()
        
        # Handle both old format (direct state_dict) and new format (dictionary)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            m_k = checkpoint['m_k']
        else:
            # Old format: just the state dict directly
            model.load_state_dict(checkpoint)
            # Need to load m_k from m_actual.npy file
            m_actual_path = os.path.join(shadow_models_dir, 'm_actual.npy')
            if os.path.exists(m_actual_path):
                all_m = np.load(m_actual_path)
                model_idx = int(model_file.split('_')[1].split('.')[0])
                m_k = all_m[model_idx]
            else:
                raise ValueError(f"Cannot find m_actual.npy for old format model: {model_file}")
        
        shadow_models.append(model)
        m_actual.append(m_k)
    
    return shadow_models, np.array(m_actual)


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
        
        print(f"Loaded configuration from checkpoint:")
        print(f"  - Attack Directory: {attack_dir}")
        print(f"  - Num Queries: {NUM_QUERIES}")
        print(f"  - Num Shadow Models: {NUM_SHADOW_MODELS}")
        print(f"  - Num Blocks: {NUM_BLOCKS}")
        print(f"  - MCMC Steps: {MCMC_STEPS}")
        print(f"  - Burn-in: {BURN_IN}")
        print(f"  - Confidence Threshold: {CONFIDENCE_THRESHOLD}")
        print(f"  - Starting from Block: {start_block}")
        print(f"  - Anchored Points: {np.sum(anchors != -1)} / {len(anchors)}")
        
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
        
        print(f"Attack configuration:")
        print(f"  - Checkpoint Directory: {checkpoint_dir}")
        print(f"  - Target Model: {target_model_path}")
        print(f"  - Num Queries: {NUM_QUERIES}")
        print(f"  - Num Shadow Models: {NUM_SHADOW_MODELS}")
        print(f"  - Num Blocks: {NUM_BLOCKS}")
        print(f"  - Confidence Threshold: {CONFIDENCE_THRESHOLD}")
        print(f"  - MCMC Steps: {MCMC_STEPS}")
        print(f"  - Burn-in: {BURN_IN}")
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
        
        # TODO fix putting this here since I hard coded to debug and not retrain
        shadow_models_dir = os.path.join(block_dir, "shadow_models")
        if os.path.exists(shadow_models_dir) and len(os.listdir(shadow_models_dir)) > 0:
            print(f"Loading existing shadow models from: {shadow_models_dir}")
            shadow_models, m_actual = load_existing_shadow_models(shadow_models_dir, NUM_SHADOW_MODELS)
        else:
            shadow_models, m_actual = train_shadow_models(
                query_indices=query_indices, 
                num_models=NUM_SHADOW_MODELS, 
                total_dataset_size=TOTAL_DATASET_SIZE,
                target_train_size=TARGET_TRAIN_SIZE,
            training_config=training_config,
            confident_memberships=anchors
        )
        

        np.save(os.path.join(block_dir, "shadow_models", "m_actual.npy"), m_actual)
        for k, model in enumerate(shadow_models):
            torch.save({
                'model_state_dict': model.state_dict(),
                'm_k': m_actual[k]
            }, os.path.join(block_dir, "shadow_models", f"shadow_{k}.pth"))
            
        C_matrices, t_bases = precompute_influence_matrices(
            shadow_models=shadow_models, 
            m_actual_matrix=m_actual, 
            query_indices=query_indices, 
            device=device
        )
        
        np.savez_compressed(
            os.path.join(block_dir, "precomputed_matrices", "influence_data.npz"), 
            C_matrices=C_matrices, 
            t_bases=t_bases
        )

        print("Evaluating target model on query points...")
        
        target_scores = evaluate_target_model(target_model_path, query_indices, device)
        
        # Calculate prior probability based on target training set size
        prior_membership_prob = TARGET_TRAIN_SIZE / TOTAL_DATASET_SIZE
        
        print(f"Running MCMC for {NUM_QUERIES} points...")
        print(f"Prior membership probability: {prior_membership_prob:.4f}")

        # Assume adversary has good prior on query
        samples = run_mcmc_block(
            target_scores=target_scores,
            C_matrices=C_matrices, 
            t_bases=t_bases, 
            m_actual=m_actual, 
            anchors=anchors, 
            num_steps=MCMC_STEPS,
            prior_prob=args.member_percentage  # TODO reinvestigate, this may note be a good assumption
        )

        
        np.save(os.path.join(block_dir, "mcmc_samples.npy"), samples)
        
        # Aggregate step
        posterior = aggregate_posterior(samples, burn_in=BURN_IN)
        
        posterior_dict = {str(q_idx): float(prob) for q_idx, prob in zip(query_indices, posterior)}
        with open(os.path.join(block_dir, "posterior_probs.json"), "w") as f:
            json.dump(posterior_dict, f, indent=2)
        
        # Print confidence table
        print(f"\n{'='*80}")
        print(f"BLOCK {block} RESULTS: Membership Predictions vs Ground Truth")
        print(f"{'='*80}")
        
        # Determine predictions based on posterior probabilities
        predictions = []
        for prob in posterior:
            if prob >= 0.99:
                predictions.append('1')  # Confident: IN
            elif prob <= 0.01:
                predictions.append('0')  # Confident: OUT
            else:
                predictions.append('?')  # Uncertain
        
        # Print table header (first 20 points)
        num_to_show = min(20, len(query_indices))
        header = "Point:      " + "  ".join([f"{i+1:>3}" for i in range(num_to_show)])
        print(header)
        print("-" * len(header))
        pred_row = "Predicted:  " + "  ".join([f"{p:>3}" for p in predictions[:num_to_show]])
        print(pred_row)
        
        truth_row = "Actual:     " + "  ".join([f"{int(gt):>3}" for gt in ground_truth[:num_to_show]])
        print(truth_row)
        
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
            
        # Cascade (new anchors for next block)
        anchors = get_new_anchors(posterior, anchors, threshold_in=CONFIDENCE_THRESHOLD, threshold_out=(1 - CONFIDENCE_THRESHOLD))

        #TODO REMOVE TO CHECK FIRST ITERATION
        exit(0)
        
        np.save(os.path.join(block_dir, "anchors_for_next_block.npy"), anchors)

if __name__ == "__main__":
    main()
