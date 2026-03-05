"""
Specificity Plotting Module
Includes functions for specificity heatmaps and comparison plots
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_specificity_heatmap_2d(importance_matrix, unique_coords, shared_coords, brain_regions,
                               anes_type, output_dir, filename_prefix='specificity_heatmap',
                               brain_regions_abbr=None, top_n=50):

    suffix = f"top_{top_n}"

    matrix_size = importance_matrix.shape[0]
    labels = [brain_regions.get(i, f'R{i}') for i in range(matrix_size)]

    unique_marker = np.zeros_like(importance_matrix, dtype=int)
    for (r, c) in unique_coords:
        unique_marker[r, c] = 1
        unique_marker[c, r] = 1

    df_unique_marker = pd.DataFrame(unique_marker, index=labels, columns=labels)
    unique_marker_path = os.path.join(output_dir, f"{filename_prefix}_unique_marker_data_{suffix}.csv")
    df_unique_marker.to_csv(unique_marker_path, encoding='utf-8-sig')

    def _plot_single_heatmap(matrix, labels_list, title_text, save_path_prefix):
        """Helper function to plot a single heatmap."""
        plt.figure(figsize=(24, 20))

        # Background heatmap
        ax = sns.heatmap(
            matrix,
            cmap=plt.cm.coolwarm,
            xticklabels=labels_list,
            yticklabels=labels_list,
            annot=False,
            cbar_kws={'label': 'Feature Importance'}
        )

        # Mark specificity connections (stars) - using annotate instead of scatter
        for (r, c) in unique_coords:
            ax.annotate('★', xy=(c + 0.5, r + 0.5),
                        ha='center', va='center',
                        color='black', fontsize=16, weight='bold')
            if r != c:
                ax.annotate('★', xy=(r + 0.5, c + 0.5),
                            ha='center', va='center',
                            color='black', fontsize=16, weight='bold')

        # Mark shared connections (dots) - using annotate instead of scatter
        for (r, c) in shared_coords:
            ax.annotate('●', xy=(c + 0.5, r + 0.5),
                        ha='center', va='center',
                        color='green', fontsize=14, weight='bold')
            if r != c:
                ax.annotate('●', xy=(r + 0.5, c + 0.5),
                            ha='center', va='center',
                            color='green', fontsize=14, weight='bold')

        plt.title(title_text, fontsize=20)
        plt.xlabel('Brain Region', fontsize=16)
        plt.ylabel('Brain Region', fontsize=16)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)

        # Save
        plt.savefig(f"{save_path_prefix}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path_prefix}.pdf", format="pdf", bbox_inches='tight')
        plt.close()

    # ====== Version 1: Full-name labels (with suffix) ======
    title_full = (
        f'{anes_type} Specificity Heatmap ({suffix.capitalize()})\n'
        f'★ = Unique to {anes_type} (Statistically Significant)\n'
        f'● = Significant but Shared with other anesthetics'
    )

    print(f"[OK] {anes_type} 2D specificity heatmap (full names, {suffix}) saved to {output_dir}")

    # ====== Version 2: Abbreviated labels (with suffix, if provided) ======
    if brain_regions_abbr is not None:
        labels_abbr = [brain_regions_abbr.get(i, f'R{i}') for i in range(matrix_size)]

        title_abbr = (
            f'{anes_type} Specificity Heatmap (Abbreviated, {suffix.capitalize()})\n'
            f'★ = Unique to {anes_type} (Statistically Significant)\n'
            f'● = Significant but Shared with other anesthetics'
        )

        print(f"[OK] {anes_type} 2D specificity heatmap (abbreviated, {suffix}) saved to {output_dir}")
    else:
        print(f"⚠️ Abbreviated labels not provided; skipping abbreviated heatmap for {anes_type}")


def plot_specificity_comparison_grid(specificity_matrices_dict, brain_regions, base_output_dir,
                                     filename_prefix='specificity_comparison_2d',
                                     title='Specificity Comparison (2D Brain Region × Brain Region)'):

    n_items = len(specificity_matrices_dict)
    if n_items == 0:
        print("⚠️ No specificity data available")
        return

    # Save raw values for all groups into summary files
    for name, data in specificity_matrices_dict.items():
        importance_matrix = data['importance_matrix']
        matrix_size = importance_matrix.shape[0]
        labels = [brain_regions.get(i, f'R{i}') for i in range(matrix_size)]

        # Save importance matrix for this group
        df_imp = pd.DataFrame(importance_matrix, index=labels, columns=labels)
        imp_path = os.path.join(base_output_dir, f"{filename_prefix}_{name}_importance_data.csv")
        df_imp.to_csv(imp_path, encoding='utf-8-sig')

        # Save marker matrices (two separate matrices)
        # 1. Unique marker matrix
        unique_marker = np.zeros_like(importance_matrix, dtype=int)
        for (r, c) in data['unique_coords']:
            unique_marker[r, c] = 1
            unique_marker[c, r] = 1

        df_unique_marker = pd.DataFrame(unique_marker, index=labels, columns=labels)
        unique_marker_path = os.path.join(base_output_dir, f"{filename_prefix}_{name}_unique_marker_data.csv")
        df_unique_marker.to_csv(unique_marker_path, encoding='utf-8-sig')

        # 2. Shared marker matrix
        shared_marker = np.zeros_like(importance_matrix, dtype=int)
        for (r, c) in data['shared_coords']:
            shared_marker[r, c] = 1
            shared_marker[c, r] = 1

        df_shared_marker = pd.DataFrame(shared_marker, index=labels, columns=labels)
        shared_marker_path = os.path.join(base_output_dir, f"{filename_prefix}_{name}_shared_marker_data.csv")
        df_shared_marker.to_csv(shared_marker_path, encoding='utf-8-sig')

    print(f"[OK] Saved all raw comparison heatmap values to {base_output_dir}")
    print("   - Each group contains: importance_data.csv, unique_marker_data.csv, shared_marker_data.csv")

    # Subplot layout
    n_cols = min(3, n_items)
    n_rows = (n_items + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    if n_items == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, data) in enumerate(specificity_matrices_dict.items()):
        ax = axes[idx]
        importance_matrix = data['importance_matrix']
        unique_coords = data['unique_coords']
        shared_coords = data['shared_coords']

        matrix_size = importance_matrix.shape[0]
        labels = [brain_regions.get(i, f'R{i}') for i in range(matrix_size)]

        # Heatmap
        sns.heatmap(
            importance_matrix,
            cmap=plt.cm.coolwarm,
            ax=ax,
            xticklabels=labels if matrix_size < 50 else False,
            yticklabels=labels if matrix_size < 50 else False,
            annot=False,
            cbar=True
        )

        # Mark unique connections (stars) - using annotate instead of scatter
        for (r, c) in unique_coords:
            ax.annotate('★', xy=(c + 0.5, r + 0.5),
                        ha='center', va='center',
                        color='black', fontsize=16, weight='bold')
            if r != c:
                ax.annotate('★', xy=(r + 0.5, c + 0.5),
                            ha='center', va='center',
                            color='black', fontsize=16, weight='bold')

        # Mark shared connections (dots) - using annotate instead of scatter
        for (r, c) in shared_coords:
            ax.annotate('●', xy=(c + 0.5, r + 0.5),
                        ha='center', va='center',
                        color='green', fontsize=14, weight='bold')
            if r != c:
                ax.annotate('●', xy=(r + 0.5, c + 0.5),
                            ha='center', va='center',
                            color='green', fontsize=14, weight='bold')

        ax.set_title(f'{name.upper()}\n★=Unique ●=Shared', fontsize=14)
        if matrix_size < 50:
            ax.tick_params(axis='x', rotation=90, labelsize=6)
            ax.tick_params(axis='y', labelsize=6)

    # Hide unused subplots
    for idx in range(n_items, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(title, fontsize=20, y=0.995)
    plt.tight_layout()

    # Save figures
    output_path = os.path.join(base_output_dir, filename_prefix)
    plt.savefig(output_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path + '.pdf', format="pdf", bbox_inches='tight')
    plt.close()

    print(f"[OK] 2D specificity comparison plot saved: {output_path}")


def plot_specificity_comparison_all_anesthetics(specificity_matrices_dict, brain_regions, base_output_dir):
    """
    Plot 2D specificity heatmap comparisons for all anesthetics (multi-panel).
    [Refactored to call the generic function]

    Args:
        specificity_matrices_dict: dict,
            {'anes_type': {'importance_matrix': array, 'unique_coords': list, 'shared_coords': list}}
        brain_regions: dict, mapping from region index to full region name
        base_output_dir: str, output directory
    """
    plot_specificity_comparison_grid(
        specificity_matrices_dict,
        brain_regions,
        base_output_dir,
        filename_prefix='all_anesthetics_specificity_comparison_2d',
        title='Anesthetics Specificity Comparison (2D Brain Region × Brain Region)'
    )


def plot_shared_matrix_heatmap(shared_matrix, shared_coords, brain_regions,
                              output_dir, brain_regions_abbr=None):
    """
    Plot a 2D heatmap of connections shared across anesthetics (shared connections matrix).

    Args:
        shared_matrix: 2D numpy array (n_regions x n_regions), shared connection matrix (1=shared, 0=not shared)
        shared_coords: list of tuples, shared connection coordinates [(row, col), ...]
        brain_regions: dict, mapping from region index to full region name
        output_dir: str, output directory path
        brain_regions_abbr: dict, mapping from region index to abbreviation (optional)
    """
    matrix_size = shared_matrix.shape[0]
    labels = [brain_regions.get(i, f'R{i}') for i in range(matrix_size)]

    # ====== Helper: plot a single heatmap ======
    def _plot_single_shared_heatmap(labels_list, title_text, save_path_prefix):
        """Helper function to plot a single shared-matrix heatmap."""
        plt.figure(figsize=(24, 20))

        # Binary heatmap (0/1)
        ax = sns.heatmap(
            shared_matrix,
            cmap='YlOrRd',  # yellow-orange-red gradient
            xticklabels=labels_list,
            yticklabels=labels_list,
            annot=False,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Shared Connection (1=Yes, 0=No)'}
        )

        # Mark shared connection positions (optional, using dots)
        for (r, c) in shared_coords:
            ax.annotate('●', xy=(c + 0.5, r + 0.5),
                        ha='center', va='center',
                        color='darkred', fontsize=14, weight='bold')
            if r != c:
                ax.annotate('●', xy=(r + 0.5, c + 0.5),
                            ha='center', va='center',
                            color='darkred', fontsize=14, weight='bold')

        plt.title(title_text, fontsize=20)
        plt.xlabel('Brain Region', fontsize=16)
        plt.ylabel('Brain Region', fontsize=16)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)

        # Save
        plt.savefig(f"{save_path_prefix}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path_prefix}.pdf", format="pdf", bbox_inches='tight')
        plt.close()

    # ====== Version 1: Full-name labels ======
    title_full = (
        "Shared Connections Across Anesthetics\n"
        "(Connections unique in ≥2 anesthetics)\n"
        "● = Shared connection"
    )
    _plot_single_shared_heatmap(
        labels,
        title_full,
        os.path.join(output_dir, "shared_connections_heatmap")
    )
    print(f"[OK] Shared connections heatmap (full names) saved to {output_dir}")

    # ====== Version 2: Abbreviated labels (if provided) ======
    if brain_regions_abbr is not None:
        labels_abbr = [brain_regions_abbr.get(i, f'R{i}') for i in range(matrix_size)]

        title_abbr = (
            "Shared Connections Across Anesthetics (Abbreviated)\n"
            "(Connections unique in ≥2 anesthetics)\n"
            "● = Shared connection"
        )
        _plot_single_shared_heatmap(
            labels_abbr,
            title_abbr,
            os.path.join(output_dir, "shared_connections_heatmap_abbr")
        )
        print(f"[OK] Shared connections heatmap (abbreviated) saved to {output_dir}")
    else:
        print("⚠️ Abbreviated labels not provided; skipping abbreviated shared heatmap")


def plot_within_receptor_comparison_grid(drug1, drug2, result1, result2, brain_regions,
                                        output_dir, receptor_name, brain_regions_abbr=None):

    matrix1 = result1['importance_matrix']
    matrix2 = result2['importance_matrix']
    unique_coords1 = result1['unique_coords']
    shared_coords1 = result1['shared_coords']
    unique_coords2 = result2['unique_coords']
    shared_coords2 = result2['shared_coords']

    matrix_size = matrix1.shape[0]
    labels = [brain_regions.get(i, f'R{i}') for i in range(matrix_size)]

    def _plot_comparison(labels_list, suffix_name):
        """Internal function to plot the comparison figure."""
        fig, axes = plt.subplots(1, 2, figsize=(48, 20))

        # Left: drug1
        sns.heatmap(
            matrix1, cmap=plt.cm.coolwarm,
            xticklabels=labels_list, yticklabels=labels_list,
            annot=False, ax=axes[0],
            cbar_kws={'label': 'Feature Importance'}
        )

        # Mark drug1 specificity - using annotate instead of scatter
        for (r, c) in unique_coords1:
            axes[0].annotate('★', xy=(c + 0.5, r + 0.5),
                             ha='center', va='center',
                             color='black', fontsize=16, weight='bold')
            if r != c:
                axes[0].annotate('★', xy=(r + 0.5, c + 0.5),
                                 ha='center', va='center',
                                 color='black', fontsize=16, weight='bold')

        for (r, c) in shared_coords1:
            axes[0].annotate('●', xy=(c + 0.5, r + 0.5),
                             ha='center', va='center',
                             color='green', fontsize=14, weight='bold')
            if r != c:
                axes[0].annotate('●', xy=(r + 0.5, c + 0.5),
                                 ha='center', va='center',
                                 color='green', fontsize=14, weight='bold')

        axes[0].set_title(f'{drug1.upper()} Specificity\n★ = Unique to {drug1.upper()}, ● = Shared', fontsize=18)
        axes[0].set_xlabel('Brain Region', fontsize=14)
        axes[0].set_ylabel('Brain Region', fontsize=14)
        axes[0].tick_params(axis='both', labelsize=8)
        plt.setp(axes[0].get_xticklabels(), rotation=90)

        # Right: drug2
        sns.heatmap(
            matrix2, cmap=plt.cm.coolwarm,
            xticklabels=labels_list, yticklabels=labels_list,
            annot=False, ax=axes[1],
            cbar_kws={'label': 'Feature Importance'}
        )

        # Mark drug2 specificity - using annotate instead of scatter
        for (r, c) in unique_coords2:
            axes[1].annotate('★', xy=(c + 0.5, r + 0.5),
                             ha='center', va='center',
                             color='black', fontsize=16, weight='bold')
            if r != c:
                axes[1].annotate('★', xy=(r + 0.5, c + 0.5),
                                 ha='center', va='center',
                                 color='black', fontsize=16, weight='bold')

        for (r, c) in shared_coords2:
            axes[1].annotate('●', xy=(c + 0.5, r + 0.5),
                             ha='center', va='center',
                             color='green', fontsize=14, weight='bold')
            if r != c:
                axes[1].annotate('●', xy=(r + 0.5, c + 0.5),
                                 ha='center', va='center',
                                 color='green', fontsize=14, weight='bold')

        axes[1].set_title(f'{drug2.upper()} Specificity\n★ = Unique to {drug2.upper()}, ● = Shared', fontsize=18)
        axes[1].set_xlabel('Brain Region', fontsize=14)
        axes[1].set_ylabel('Brain Region', fontsize=14)
        axes[1].tick_params(axis='both', labelsize=8)
        plt.setp(axes[1].get_xticklabels(), rotation=90)

        fig.suptitle(
            f'{receptor_name} Receptor: {drug1.upper()} vs {drug2.upper()} Specificity Comparison',
            fontsize=22, y=0.98
        )

        plt.tight_layout()

        # Save
        save_prefix = os.path.join(output_dir, f"{receptor_name}_comparison_{suffix_name}")
        plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_prefix}.pdf", format="pdf", bbox_inches='tight')
        plt.close()

        print(f"[OK] {receptor_name} receptor comparison plot saved ({suffix_name}): {save_prefix}")

    # Full-name version
    _plot_comparison(labels, "full")

    # Abbreviated version (if provided)
    if brain_regions_abbr is not None:
        labels_abbr = [brain_regions_abbr.get(i, f'R{i}') for i in range(matrix_size)]
        _plot_comparison(labels_abbr, "abbr")