"""Pairwise ranking accuracy metric for AV-Sync evaluation.

Given predicted scores per (sample, method) and a pairs file, for every pair
with a clear ground-truth preference we check whether the predicted ordering
matches. Pairs are bucketed by GT-score gap into easy / medium / hard.
"""
from collections import defaultdict


def compute_pair_accuracy(score_map, pairs):
    """
    Args:
        score_map: dict {(sample_name, method): {"pred_score": float, "gt_score": float}}
        pairs:     dict {sample_name: [[method1, method2, ...], ...]}
    Returns:
        dict with overall accuracy and per-difficulty breakdown.
    """
    total_pairs = 0
    correct_pairs = 0
    missing_pairs = 0

    difficulty_stats = {
        "easy": {"total": 0, "correct": 0},    # gt_diff > 0.5
        "medium": {"total": 0, "correct": 0},  # 0.1 < gt_diff <= 0.5
        "hard": {"total": 0, "correct": 0},    # gt_diff <= 0.1
    }
    method_pair_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for sample_name, pair_list in pairs.items():
        for pair in pair_list:
            method1, method2 = pair[0], pair[1]
            key1, key2 = (sample_name, method1), (sample_name, method2)
            if key1 not in score_map or key2 not in score_map:
                missing_pairs += 1
                continue

            gt1 = score_map[key1]["gt_score"]
            gt2 = score_map[key2]["gt_score"]
            if gt1 == gt2:
                continue  # no clear preference

            pred1 = score_map[key1]["pred_score"]
            pred2 = score_map[key2]["pred_score"]

            total_pairs += 1
            is_correct = (pred1 > pred2) == (gt1 > gt2)
            if is_correct:
                correct_pairs += 1

            gt_diff = abs(gt1 - gt2)
            difficulty = "easy" if gt_diff > 0.5 else "medium" if gt_diff > 0.1 else "hard"
            difficulty_stats[difficulty]["total"] += 1
            if is_correct:
                difficulty_stats[difficulty]["correct"] += 1

            mp_key = tuple(sorted([method1, method2]))
            method_pair_stats[mp_key]["total"] += 1
            if is_correct:
                method_pair_stats[mp_key]["correct"] += 1

    return {
        "total_pairs": total_pairs,
        "correct_pairs": correct_pairs,
        "missing_pairs": missing_pairs,
        "accuracy": correct_pairs / total_pairs if total_pairs > 0 else 0.0,
        "difficulty_stats": difficulty_stats,
        "method_pair_stats": dict(method_pair_stats),
    }
