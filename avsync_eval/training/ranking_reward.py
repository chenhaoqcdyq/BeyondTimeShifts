"""
Ranking-based Reward Functions for RL Training.

Computes global ranking quality rewards (NDCG, Kendall Tau, Spearman Rho, 
Top-1 Accuracy, MRR) based on predicted scores vs. ground truth scores.

These rewards bridge the gap between pairwise training signals and the 
global ranking evaluation metrics used in Eval_RankAcc.py.
"""

import torch
import numpy as np
from scipy.stats import kendalltau, spearmanr
from typing import Dict, Optional


def compute_ndcg(pred_scores: torch.Tensor, gt_scores: torch.Tensor) -> torch.Tensor:
    """
    Compute NDCG (Normalized Discounted Cumulative Gain) for a batch.
    
    Args:
        pred_scores: (B, K) predicted scores for K methods
        gt_scores:   (B, K) ground truth scores for K methods
    
    Returns:
        (B,) NDCG values per sample
    """
    B, K = pred_scores.shape
    ndcg_values = []
    
    for b in range(B):
        ps = pred_scores[b].detach().cpu().float().numpy()
        gs = gt_scores[b].detach().cpu().float().numpy()
        
        # Get predicted ranking order (descending by pred score)
        pred_order = np.argsort(-ps)
        
        # Relevance scores: use GT scores as relevance, placed in predicted order
        pred_rel = gs[pred_order]
        
        # Ideal order: GT scores sorted descending
        ideal_rel = np.sort(gs)[::-1]
        
        # DCG formula: sum(rel_i / log2(i+2))
        discounts = np.log2(np.arange(K) + 2)
        dcg = np.sum(pred_rel / discounts)
        idcg = np.sum(ideal_rel / discounts)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_values.append(ndcg)
    
    return torch.tensor(ndcg_values, dtype=pred_scores.dtype, device=pred_scores.device)


def compute_kendall_tau(pred_scores: torch.Tensor, gt_scores: torch.Tensor) -> torch.Tensor:
    """
    Compute Kendall's Tau correlation for a batch.
    
    Args:
        pred_scores: (B, K) predicted scores
        gt_scores:   (B, K) ground truth scores
    
    Returns:
        (B,) Kendall Tau values per sample, in range [-1, 1]
    """
    B, K = pred_scores.shape
    tau_values = []
    
    for b in range(B):
        ps = pred_scores[b].detach().cpu().float().numpy()
        gs = gt_scores[b].detach().cpu().float().numpy()
        
        if K < 2:
            tau_values.append(0.0)
            continue
        
        tau, _ = kendalltau(ps, gs)
        tau_values.append(tau if not np.isnan(tau) else 0.0)
    
    return torch.tensor(tau_values, dtype=pred_scores.dtype, device=pred_scores.device)


def compute_spearman_rho(pred_scores: torch.Tensor, gt_scores: torch.Tensor) -> torch.Tensor:
    """
    Compute Spearman's Rho correlation for a batch.
    
    Args:
        pred_scores: (B, K) predicted scores
        gt_scores:   (B, K) ground truth scores
    
    Returns:
        (B,) Spearman Rho values per sample, in range [-1, 1]
    """
    B, K = pred_scores.shape
    rho_values = []
    
    for b in range(B):
        ps = pred_scores[b].detach().cpu().float().numpy()
        gs = gt_scores[b].detach().cpu().float().numpy()
        
        if K < 2:
            rho_values.append(0.0)
            continue
        
        rho, _ = spearmanr(ps, gs)
        rho_values.append(rho if not np.isnan(rho) else 0.0)
    
    return torch.tensor(rho_values, dtype=pred_scores.dtype, device=pred_scores.device)


def compute_top1_acc(pred_scores: torch.Tensor, gt_scores: torch.Tensor) -> torch.Tensor:
    """
    Compute Top-1 Accuracy: does the predicted best method match the GT best?
    
    Args:
        pred_scores: (B, K)
        gt_scores:   (B, K)
    
    Returns:
        (B,) binary accuracy (0 or 1)
    """
    pred_best = pred_scores.argmax(dim=1)
    gt_best = gt_scores.argmax(dim=1)
    return (pred_best == gt_best).float()


def compute_mrr(pred_scores: torch.Tensor, gt_scores: torch.Tensor) -> torch.Tensor:
    """
    Compute MRR (Mean Reciprocal Rank): how quickly we find the true best item.
    
    Args:
        pred_scores: (B, K)
        gt_scores:   (B, K)
    
    Returns:
        (B,) MRR values per sample
    """
    B, K = pred_scores.shape
    mrr_values = []
    
    for b in range(B):
        ps = pred_scores[b].detach().cpu().float().numpy()
        gs = gt_scores[b].detach().cpu().float().numpy()
        
        gt_best_idx = np.argmax(gs)
        pred_order = np.argsort(-ps)  # descending
        
        # Find where the GT best item appears in predicted ranking
        rank = np.where(pred_order == gt_best_idx)[0]
        if len(rank) > 0:
            mrr_values.append(1.0 / (rank[0] + 1))
        else:
            mrr_values.append(0.0)
    
    return torch.tensor(mrr_values, dtype=pred_scores.dtype, device=pred_scores.device)


def compute_pairwise_concordance(pred_scores: torch.Tensor, gt_scores: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise concordance rate: fraction of all C(K,2) pairs 
    where predicted ordering matches GT ordering.
    This is directly related to the pair accuracy metric.
    
    Args:
        pred_scores: (B, K)
        gt_scores:   (B, K)
    
    Returns:
        (B,) concordance rates in [0, 1]
    """
    B, K = pred_scores.shape
    concordance = []
    
    for b in range(B):
        ps = pred_scores[b]
        gs = gt_scores[b]
        
        total, correct = 0, 0
        for i in range(K):
            for j in range(i + 1, K):
                # Skip if GT scores are equal
                if gs[i] == gs[j]:
                    continue
                total += 1
                # Check if predicted order matches GT order
                pred_order = (ps[i] > ps[j])
                gt_order = (gs[i] > gs[j])
                if pred_order == gt_order:
                    correct += 1
        
        concordance.append(correct / total if total > 0 else 0.5)
    
    return torch.tensor(concordance, dtype=pred_scores.dtype, device=pred_scores.device)


def compute_ranking_reward(
    pred_scores: torch.Tensor,
    gt_scores: torch.Tensor,
    weights: Optional[Dict[str, float]] = None,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute a composite ranking reward from multiple metrics.
    
    Args:
        pred_scores: (B, K) predicted scores for K methods
        gt_scores:   (B, K) ground truth scores for K methods
        weights:     Dict of metric_name -> weight. Default:
                     {'ndcg': 0.3, 'kendall': 0.2, 'spearman': 0.2, 
                      'top1': 0.15, 'pairwise': 0.15}
        mask:        Optional (B, K) mask for valid methods (1=valid, 0=padding)
    
    Returns:
        Dict with:
        - 'reward': (B,) composite reward per sample
        - 'ndcg': (B,) NDCG values
        - 'kendall': (B,) Kendall Tau values
        - 'spearman': (B,) Spearman Rho values
        - 'top1': (B,) Top-1 accuracy
        - 'pairwise': (B,) Pairwise concordance rate
        - 'mrr': (B,) MRR values
    """
    if weights is None:
        weights = {
            'ndcg': 0.25,
            'kendall': 0.20,
            'spearman': 0.20,
            'top1': 0.15,
            'pairwise': 0.10,
            'mrr': 0.10,
        }
    
    # If mask is provided, only use valid entries
    if mask is not None:
        # Replace padding with very low scores to not affect ranking
        pred_scores = pred_scores.clone()
        gt_scores = gt_scores.clone()
        pred_scores[mask == 0] = -1e9
        gt_scores[mask == 0] = -1e9
    
    # Compute individual metrics
    ndcg = compute_ndcg(pred_scores, gt_scores)
    kendall = compute_kendall_tau(pred_scores, gt_scores)
    spearman = compute_spearman_rho(pred_scores, gt_scores)
    top1 = compute_top1_acc(pred_scores, gt_scores)
    pairwise = compute_pairwise_concordance(pred_scores, gt_scores)
    mrr = compute_mrr(pred_scores, gt_scores)
    
    # Normalize Kendall and Spearman from [-1, 1] to [0, 1]
    kendall_norm = (kendall + 1.0) / 2.0
    spearman_norm = (spearman + 1.0) / 2.0
    
    # Composite reward
    reward = (
        weights.get('ndcg', 0) * ndcg +
        weights.get('kendall', 0) * kendall_norm +
        weights.get('spearman', 0) * spearman_norm +
        weights.get('top1', 0) * top1 +
        weights.get('pairwise', 0) * pairwise +
        weights.get('mrr', 0) * mrr
    )
    
    return {
        'reward': reward,
        'ndcg': ndcg,
        'kendall': kendall,
        'spearman': spearman,
        'top1': top1,
        'pairwise': pairwise,
        'mrr': mrr,
    }


def compute_ranking_reward_for_rollouts(
    rollout_scores: torch.Tensor,
    gt_scores: torch.Tensor,
    weights: Optional[Dict[str, float]] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute ranking rewards for GRPO rollouts.
    
    Args:
        rollout_scores: (B, K, num_rollout) - rollout scores for K methods, num_rollout rollouts
        gt_scores:      (B, K) - ground truth scores
        weights:        metric weights
        mask:           (B, K) validity mask
    
    Returns:
        (B, num_rollout) reward tensor for each rollout
    """
    B, K, num_rollout = rollout_scores.shape
    rewards = torch.zeros(B, num_rollout, device=rollout_scores.device, dtype=rollout_scores.dtype)
    
    for r in range(num_rollout):
        # Extract scores for this rollout: (B, K)
        rollout_r = rollout_scores[:, :, r]
        result = compute_ranking_reward(rollout_r, gt_scores, weights, mask)
        rewards[:, r] = result['reward']
    
    return rewards
