"""
Statistical Testing Module
Includes statistical functions such as Kruskal–Wallis, Dunn’s test, Mann–Whitney U test, etc.
"""

import pandas as pd
import numpy as np
from scipy.stats import kruskal, mannwhitneyu
from scikit_posthocs import posthoc_dunn


def get_top_indices_2d(contrib_matrix, top_n=50):
    """
    Get the coordinates of the top-N largest values in a 2D contribution matrix.

    Args:
        contrib_matrix: 2D numpy array or pandas DataFrame, functional connectivity contribution matrix.
        top_n: int or str, number of top connections to keep (default: 50).
               If a string (e.g., 'top_20', 'top_50'), the number will be extracted.

    Returns:
        list of tuples: [(row_idx, col_idx), ...], coordinates of the top-N contribution values.
    """
    # Handle top_n parameter (support string formats like 'top_20')
    if isinstance(top_n, str):
        # Extract digits from the string, e.g., 'top_20' -> 20
        import re
        match = re.search(r'\d+', top_n)
        if match:
            top_n = int(match.group())
        else:
            raise ValueError(
                f"Cannot extract a number from string '{top_n}'. "
                f"Please use the 'top_N' format (e.g., 'top_20', 'top_50') or pass an integer directly. "
                f"If you want to use all connections, convert 'none' to an integer using parse_threshold_mode first."
            )

    # Ensure top_n is an integer
    top_n = int(top_n)

    # Convert to numpy array
    if isinstance(contrib_matrix, pd.DataFrame):
        matrix = contrib_matrix.values
    else:
        matrix = contrib_matrix

    # Get matrix shape
    n_regions = matrix.shape[0]

    # Only consider the upper triangle (excluding the diagonal) since FC matrices are symmetric
    indices_list = []
    values_list = []

    for i in range(n_regions):
        for j in range(i + 1, n_regions):  # upper triangle only
            indices_list.append((i, j))
            values_list.append(np.abs(matrix[i, j]))  # take absolute value

    # Sort by value and take top_n
    values_array = np.array(values_list)
    sorted_indices = np.argsort(values_array)[::-1][:top_n]

    top_coords = [indices_list[idx] for idx in sorted_indices]

    return top_coords


def perform_kruskal(group_data_list, alpha=0.05):
    """
    Perform the Kruskal–Wallis H test (non-parametric test).

    Args:
        group_data_list: list of arrays, data for each group.
        alpha: float, significance level (default: 0.05).

    Returns:
        dict: {'stat': test statistic, 'p_val': p-value, 'significant': whether significant}.
    """
    # Check whether all values are identical (no variability)
    all_values = np.concatenate([np.asarray(g).flatten() for g in group_data_list])

    # If all values are identical, the test cannot be performed; return a non-significant result
    if len(np.unique(all_values)) == 1:
        # print(f"[WARNING] Kruskal–Wallis test skipped: all values are identical (unique value={all_values[0]:.6f}). "
        #       f"Returning p=1.0 (no significant difference).")
        return {
            'stat': 0.0,
            'p_val': 1.0,  # p=1 indicates no significant difference
            'significant': False
        }

    try:
        kruskal_stat, kruskal_p_val = kruskal(*group_data_list)
    except ValueError as e:
        # Catch other possible errors (e.g., too-small sample size)
        print(f"{e}")
        return {
            'stat': 0.0,
            'p_val': 1.0,
            'significant': False
        }

    # Ensure scalar outputs (handle possible array outputs)
    if isinstance(kruskal_stat, np.ndarray):
        kruskal_stat = kruskal_stat.item() if kruskal_stat.size == 1 else kruskal_stat[0]
    if isinstance(kruskal_p_val, np.ndarray):
        kruskal_p_val = kruskal_p_val.item() if kruskal_p_val.size == 1 else kruskal_p_val[0]

    return {
        'stat': float(kruskal_stat),
        'p_val': float(kruskal_p_val),
        'significant': bool(kruskal_p_val < alpha)
    }


def perform_mann_whitney(group1_data, group2_data, alpha=0.05):
    """
    [Kept but currently unused] Perform the Mann–Whitney U test (non-parametric test for two-group comparisons).

    Note:
        This project currently standardizes on Kruskal–Wallis + Dunn’s test (including two-group comparisons).
        This function is kept as a backup, but it has been commented out in specificity_analysis.py.

    Args:
        group1_data: array, data for group 1.
        group2_data: array, data for group 2.
        alpha: float, significance level (default: 0.05).

    Returns:
        dict: {'stat': test statistic, 'p_val': p-value, 'significant': whether significant}.
    """
    # Two-sided test
    mann_whitney_stat, mann_whitney_p_val = mannwhitneyu(
        group1_data, group2_data, alternative='two-sided'
    )

    # Ensure scalar outputs
    if isinstance(mann_whitney_stat, np.ndarray):
        mann_whitney_stat = mann_whitney_stat.item() if mann_whitney_stat.size == 1 else mann_whitney_stat[0]
    if isinstance(mann_whitney_p_val, np.ndarray):
        mann_whitney_p_val = mann_whitney_p_val.item() if mann_whitney_p_val.size == 1 else mann_whitney_p_val[0]

    return {
        'stat': float(mann_whitney_stat),
        'p_val': float(mann_whitney_p_val),
        'significant': bool(mann_whitney_p_val < alpha)
    }


def check_is_unique(group_data_list, group_idx, valid_groups, alpha=0.05):
    """
    Check whether a given group is a specificity group (i.e., significantly different from all other groups).
    Uses Dunn’s post-hoc test with Bonferroni correction.

    Args:
        group_data_list: list of arrays, data for each group.
        group_idx: int, index of the target group.
        valid_groups: array, list/array of indices for valid groups.
        alpha: float, significance level (default: 0.05).

    Returns:
        dict: {'is_unique': bool, 'dunn_pvalues': list of float, 'min_pvalue': float}.
    """
    # Convert data to a DataFrame format required by posthoc_dunn
    # Build a DataFrame with 'values' and 'groups' columns
    data_for_dunn = []
    group_labels = []

    for i, group_data in enumerate(group_data_list):
        # Ensure data is a 1D array
        if isinstance(group_data, np.ndarray):
            values = group_data.flatten()
        else:
            values = np.array(group_data).flatten()

        # Append values and corresponding group labels
        data_for_dunn.extend(values.tolist())
        group_labels.extend([i] * len(values))

    # Create DataFrame
    df_dunn = pd.DataFrame({
        'values': data_for_dunn,
        'groups': group_labels
    })

    # Perform Dunn’s post-hoc test (pairwise comparisons)
    dunn_results = posthoc_dunn(df_dunn, val_col='values', group_col='groups', p_adjust='bonferroni')

    k = len(group_data_list)

    # Find the position of the target group within the valid group list
    group_pos_in_valid = np.where(valid_groups == group_idx)[0]
    if len(group_pos_in_valid) == 0:
        return {'is_unique': False, 'dunn_pvalues': [], 'min_pvalue': 1.0}

    group_pos = group_pos_in_valid[0]

    # Collect p-values against other groups
    dunn_pvalues = []
    is_unique = True

    # Check whether this group is significantly different from all other groups
    # dunn_results is a symmetric matrix with group labels as both rows and columns
    for j in range(k):
        if j != group_pos:
            p_val = float(dunn_results.iloc[group_pos, j])
            dunn_pvalues.append(p_val)
            if p_val >= alpha:  # not significant
                is_unique = False

    min_pvalue = min(dunn_pvalues) if dunn_pvalues else 1.0

    return {
        'is_unique': is_unique,
        'dunn_pvalues': dunn_pvalues,
        'min_pvalue': min_pvalue
    }