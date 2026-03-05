import os
import numpy as np
from .heatmap import plot_feature_importance


def visualize_all_anesthetics_comparison(
    all_feature_importances,
    all_anesthetic_labels,
    x_axis_labels,
    base_output_dir,
    N=20
):
    """
    Generate a region-wise comparison heatmap across all anesthetics.

    Parameters:
        all_feature_importances: list, each element is a region-importance vector for one anesthetic
        all_anesthetic_labels: list, anesthetic names (UPPERCASE)
        x_axis_labels: list of region labels (x-axis labels)
        base_output_dir: output directory
        N: number of Top-N regions to annotate
    """
    if all_feature_importances and x_axis_labels:
        stacked_importances = np.stack(all_feature_importances, axis=0)
        save_name = os.path.join(
            base_output_dir,
            'ALL_ANESTHETICS_REGION_comparison_heatmap'
        )
        plot_feature_importance(
            stacked_importances,
            index_names=all_anesthetic_labels,
            label_names=x_axis_labels,
            save_name=save_name,
            N=N,
            mode='topN_and_greater_than_zero'
        )
        print("✅ Region-wise comparison heatmap for all anesthetics has been generated.")
    else:
        print("⚠️ Skipping heatmap generation: incomplete data.")