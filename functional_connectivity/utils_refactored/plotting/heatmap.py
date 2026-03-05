"""
Heatmap Plotting Module
Includes feature-importance heatmaps and basic heatmap plotting functions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(feature_importances, index_names, label_names, save_name, N, mode='topN_and_greater_than_zero'):
    """
    Plot a feature-importance heatmap and mark important regions based on the selected mode.

    Args:
        feature_importances: Feature-importance matrix.
        index_names: Row labels (anesthetic names).
        label_names: Column labels (brain region names).
        save_name: Output file prefix (full path without extension).
        N: Top-N value.
        mode: Marking mode ('greater_than_zero' or 'topN_and_greater_than_zero').
    """
    row_min = feature_importances.min(axis=1, keepdims=True)
    row_max = feature_importances.max(axis=1, keepdims=True)
    feature_importances = (feature_importances - row_min) / (row_max - row_min + 1e-9)

    num = feature_importances.shape[0]
    plot_width = max(20, len(label_names) * 0.2)
    plot_height = max(10, num * 0.5)

    plt.figure(figsize=(plot_width, plot_height))

    df = pd.DataFrame(feature_importances, index=index_names, columns=label_names)
    df.to_excel(save_name + '.xlsx')

    sns.heatmap(df, annot=False, cmap=plt.cm.coolwarm)

    if N > 0:
        if mode == 'greater_than_zero':
            for i in range(num):
                for j in range(feature_importances.shape[1]):
                    if feature_importances[i, j] > 0:
                        plt.scatter(j + 0.5, i + 0.5, color='black', marker='*', s=20)
        elif mode == 'topN_and_greater_than_zero' and N > 0:
            actual_N = min(N, feature_importances.shape[1])
            top_N_indices = np.argsort(feature_importances, axis=1)[:, -actual_N:]

            for i in range(num):
                for j in top_N_indices[i]:
                    if feature_importances[i, j] > 0:
                        plt.scatter(j + 0.5, i + 0.5, color='black', marker='*', s=20)

    plt.xlabel('Brain Regions (Features)')
    plt.ylabel('Anesthetics')
    plt.title('Brain Region Contribution Heatmap (All Anesthetics)')

    if len(label_names) < 200:
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.5)
    else:
        plt.xticks([])
        plt.xlabel(f'Brain Regions ({len(label_names)} features)')
        plt.subplots_adjust(bottom=0.1)

    plt.savefig(save_name + '.pdf', format="pdf", bbox_inches='tight')
    plt.savefig(save_name + '.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_importance_heatmap(matrix, axis_labels, title, save_prefix, axis_labels_abbr=None, threshold_mode='top_50'):
    """
    Plot a feature-importance matrix heatmap for a single anesthetic (brain region x brain region),
    and save it as PNG, PDF, and the raw values (CSV/Excel).

    Four versions are generated:
      1) Full-name version (all values)
      2) Abbreviated-label version (all values, if abbreviations are provided)
      3) Full-name Top-N version
      4) Abbreviated-label Top-N version (if abbreviations are provided)

    Args:
        matrix: 2D numpy array (square matrix).
        axis_labels: List of full brain-region names, length = matrix.shape[0].
        title: Figure title.
        save_prefix: Output file prefix (path without extension).
        axis_labels_abbr: Optional list of abbreviated brain-region labels. If None, abbreviated versions are skipped.
        threshold_mode: str, selection mode (e.g., 'top_10', 'top_20', 'top_50'), default 'top_50'.
    """
    try:
        # Parse threshold_mode to get Top-N
        import re
        match = re.search(r'top[_-]?(\d+)', threshold_mode, re.IGNORECASE)
        if match:
            top_n = int(match.group(1))
        else:
            top_n = 50  # default

        topn_suffix = f"_top{top_n}"  # dynamic filename suffix

        # Save raw numeric data (with labels)
        df_matrix = pd.DataFrame(matrix, index=axis_labels, columns=axis_labels)
        df_matrix.to_csv(save_prefix + "_data.csv", encoding='utf-8-sig')
        print(f"✅ Saved raw heatmap values: {save_prefix}_data.csv")

        # ========== Prepare Top-N masked matrix ==========
        # Get indices of the top-N largest values in the upper triangle
        matrix_size = matrix.shape[0]
        upper_tri_indices = np.triu_indices(matrix_size, k=1)
        upper_tri_values = matrix[upper_tri_indices]
        top_n_indices = np.argsort(upper_tri_values)[-top_n:]  # indices of top-N largest values

        # Create a masked matrix (set values outside Top-N to 0)
        matrix_topn = np.zeros_like(matrix)
        for idx in top_n_indices:
            r = upper_tri_indices[0][idx]
            c = upper_tri_indices[1][idx]
            matrix_topn[r, c] = matrix[r, c]
            matrix_topn[c, r] = matrix[r, c]  # symmetric

        # Save Top-N numeric data
        df_matrix_topn = pd.DataFrame(matrix_topn, index=axis_labels, columns=axis_labels)
        df_matrix_topn.to_csv(save_prefix + f"_data{topn_suffix}.csv", encoding='utf-8-sig')
        print(f"✅ Saved Top{top_n} raw heatmap values: {save_prefix}_data{topn_suffix}.csv")

        # ====== Version 1: Full-name heatmap ======
        plt.figure(figsize=(40, 32))
        ax = sns.heatmap(
            matrix,
            cmap=plt.cm.coolwarm,
            annot=False,
            xticklabels=axis_labels,
            yticklabels=axis_labels
        )
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation=0, va='center')
        plt.title(title, fontsize=40)
        plt.xlabel('Brain Region', fontsize=30)
        plt.ylabel('Brain Region', fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Save full-name version
        plt.savefig(save_prefix + ".png", dpi=300, bbox_inches='tight')
        plt.savefig(save_prefix + ".pdf", format="pdf", bbox_inches='tight')
        plt.close()
        print(f"✅ Saved full-name heatmap: {save_prefix}.png / .pdf")

        # ====== Version 2: Full-name Top-N heatmap ======
        plt.figure(figsize=(40, 32))
        ax = sns.heatmap(
            matrix_topn,
            cmap=plt.cm.coolwarm,
            annot=False,
            xticklabels=axis_labels,
            yticklabels=axis_labels
        )
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation=0, va='center')
        plt.title(title + f" (Top{top_n} Connections)", fontsize=40)
        plt.xlabel('Brain Region', fontsize=30)
        plt.ylabel('Brain Region', fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Save full-name Top-N version
        plt.savefig(save_prefix + f"{topn_suffix}.png", dpi=300, bbox_inches='tight')
        plt.savefig(save_prefix + f"{topn_suffix}.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        print(f"✅ Saved full-name Top{top_n} heatmap: {save_prefix}{topn_suffix}.png / .pdf")

        # ====== Versions 3 & 4: Abbreviated versions (if provided) ======
        if axis_labels_abbr is not None:
            # Version 3: Abbreviated heatmap (all values)
            plt.figure(figsize=(40, 32))
            ax = sns.heatmap(
                matrix,
                cmap=plt.cm.coolwarm,
                annot=False,
                xticklabels=axis_labels_abbr,
                yticklabels=axis_labels_abbr
            )
            plt.setp(ax.get_xticklabels(), rotation=90, ha='right', rotation_mode='anchor')
            plt.setp(ax.get_yticklabels(), rotation=0, va='center')
            plt.title(title + " (Abbreviated)", fontsize=40)
            plt.xlabel('Brain Region', fontsize=30)
            plt.ylabel('Brain Region', fontsize=30)
            ax.tick_params(axis='both', which='major', labelsize=10)

            # Save abbreviated version
            plt.savefig(save_prefix + "_abbr.png", dpi=300, bbox_inches='tight')
            plt.savefig(save_prefix + "_abbr.pdf", format="pdf", bbox_inches='tight')
            plt.close()
            print(f"✅ Saved abbreviated heatmap: {save_prefix}_abbr.png / .pdf")

            # Version 4: Abbreviated Top-N heatmap
            plt.figure(figsize=(40, 32))
            ax = sns.heatmap(
                matrix_topn,
                cmap=plt.cm.coolwarm,
                annot=False,
                xticklabels=axis_labels_abbr,
                yticklabels=axis_labels_abbr
            )
            plt.setp(ax.get_xticklabels(), rotation=90, ha='right', rotation_mode='anchor')
            plt.setp(ax.get_yticklabels(), rotation=0, va='center')
            plt.title(title + f" (Abbreviated, Top{top_n} Connections)", fontsize=40)
            plt.xlabel('Brain Region', fontsize=30)
            plt.ylabel('Brain Region', fontsize=30)
            ax.tick_params(axis='both', which='major', labelsize=10)

            # Save abbreviated Top-N version
            plt.savefig(save_prefix + f"_abbr{topn_suffix}.png", dpi=300, bbox_inches='tight')
            plt.savefig(save_prefix + f"_abbr{topn_suffix}.pdf", format="pdf", bbox_inches='tight')
            plt.close()
            print(f"✅ Saved abbreviated Top{top_n} heatmap: {save_prefix}_abbr{topn_suffix}.png / .pdf")

            # Save abbreviation mapping table
            abbr_mapping = pd.DataFrame({
                'Full_Name': axis_labels,
                'Abbreviated': axis_labels_abbr
            })
            abbr_mapping.to_csv(save_prefix + "_abbreviation_mapping.csv", index=False, encoding='utf-8-sig')
            print(f"✅ Saved abbreviation mapping table: {save_prefix}_abbreviation_mapping.csv")
        else:
            print("⚠️ No abbreviated labels provided; skipping abbreviated heatmap generation")

    except Exception as e:
        print(f"Error while plotting heatmap: {e}")
        plt.close()