"""
Matrix operation module
Includes matrix processing and data loading functions
"""

import os
import pandas as pd
import numpy as np
from ..plotting.heatmap import plot_importance_heatmap


def compute_region_importance_from_matrix(importance_matrix):

    # Add each edge's importance to both connected nodes, then divide by 2 to avoid double counting
    region_importances = np.sum(importance_matrix, axis=0) / 2.0
    return region_importances


def load_and_visualize_existing_matrix(
    matrix_csv_path,
    brain_regions,
    anes_type,
    output_dir,
    brain_regions_abbr=None,
    threshold_mode='top_50'
):

    imp_matrix = pd.read_csv(matrix_csv_path, header=None).values
    n_regions = imp_matrix.shape[0]

    # Compute aggregated region-level importance
    region_importances = compute_region_importance_from_matrix(imp_matrix)

    # Generate brain region labels (full names)
    x_axis_labels = [brain_regions.get(i, f'R{i}') for i in range(n_regions)]

    # Generate brain region labels (abbreviations)
    x_axis_labels_abbr = None
    if brain_regions_abbr is not None:
        x_axis_labels_abbr = [brain_regions_abbr.get(i, f'R{i}') for i in range(n_regions)]

    # Plot heatmaps (full-name version and abbreviation version)
    title = f"{anes_type.upper()} RF Feature Importance Matrix"
    save_prefix = os.path.join(output_dir, f"{anes_type}_rf_feature_importance_matrix")
    plot_importance_heatmap(
        imp_matrix,
        x_axis_labels,
        title,
        save_prefix,
        x_axis_labels_abbr,
        threshold_mode
    )

    return region_importances, x_axis_labels