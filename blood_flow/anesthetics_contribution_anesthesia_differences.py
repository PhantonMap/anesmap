import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns
from scikit_posthocs import posthoc_dunn
from scipy.stats import kruskal

from uutils import filter_merge_and_remap_labels


def perform_kruskal(group_data_list, alpha):
    kruskal_stat, kruskal_p_val = kruskal(*group_data_list)
    print(f"perform_kruskal: {kruskal_p_val}")
    return {
        'stat': kruskal_stat,
        'p_val': kruskal_p_val,
        'significant': kruskal_p_val < alpha
    }


def check_unique_kruskal(group_data_list, group_idx, valid_groups, alpha):
    dunn_results = posthoc_dunn(group_data_list, p_adjust='Bonferroni')
    k = len(group_data_list)
    group_pos_in_valid = np.where(valid_groups == group_idx)[0]
    if len(group_pos_in_valid) == 0:
        return False
    group_pos = group_pos_in_valid[0]
    return all(dunn_results.iloc[group_pos, j] < alpha for j in range(k) if j != group_pos)


def get_top_indices(group_contrib, threshold_mode):
    if 'top_' in threshold_mode:
        top_n_value = int(threshold_mode.split('_')[1])
        return np.argsort(np.abs(group_contrib))[::-1][:top_n_value]
    elif threshold_mode == 'none':
        return np.arange(len(group_contrib))
    else:
        group_abs_contrib = np.abs(group_contrib)
        group_threshold = np.median(group_abs_contrib)
        return np.where(group_abs_contrib > group_threshold)[0]


def find_unique_regions(contrib_matrix, anesthetic_names, all_anesthetic_data, threshold_mode, alpha=0.05):
    n_groups, n_regions = contrib_matrix.shape
    unique_regions = {name: [] for name in anesthetic_names}
    all_p_values = np.zeros(n_regions)

    for group_idx in range(n_groups):
        group_name = anesthetic_names[group_idx]
        group_contrib = contrib_matrix.loc[group_name]
        top_indices = get_top_indices(group_contrib, threshold_mode)

        for region_idx in top_indices:
            values, labels = [], []
            for g, gname in enumerate(anesthetic_names):
                current_region_data = all_anesthetic_data[gname][:, region_idx]
                if current_region_data.size == 0:
                    continue
                values.extend(current_region_data.flatten())
                labels.extend([g] * current_region_data.shape[0])

            if len(labels) == 0:
                continue

            labels_np = np.array(labels)
            values_np = np.array(values)
            valid_groups = np.unique(labels_np)
            group_data_list = [values_np[labels_np == g] for g in valid_groups]
            group_data_list = [data for data in group_data_list if data.size > 0]

            if len(group_data_list) < 2:
                continue

            test_method = 'kruskal'
            print('Perform Kruskal-Wallis + Dunn')
            kruskal_result = perform_kruskal(group_data_list, alpha)
            all_p_values[region_idx] = kruskal_result['p_val']
            significant = kruskal_result['significant']
            if significant:
                is_unique = check_unique_kruskal(group_data_list, group_idx, valid_groups, alpha)

            if significant and is_unique and region_idx not in unique_regions[group_name]:
                unique_regions[group_name].append(region_idx)

            # threshold_subdir = os.path.join(result_root, 'stat_results', threshold_mode, test_method)
            # os.makedirs(threshold_subdir, exist_ok=True)
            # save_kruskal_result(threshold_subdir, region_idx, kruskal_result)

    corrected_p_values = all_p_values
    return unique_regions, corrected_p_values


def plot_contrib_heatmap(contrib_matrix, unique_regions, region_abbr, anesthetic_names, save_name=None):
    if contrib_matrix.size == 0:
        print("The contribution matrix is empty, so a heatmap cannot be plotted.")
        return

    region_count = {}
    for name, indices in unique_regions.items():
        for idx in indices:
            region_count[idx] = region_count.get(idx, 0) + 1

    unique_brain_regions = {name: [] for name in anesthetic_names}
    shared_brain_regions = []

    fig, ax = plt.subplots(figsize=(40, 15))
    cmap = plt.cm.coolwarm
    cmap.set_bad(color='gray')
    sns.heatmap(
        contrib_matrix,
        cmap=cmap,
        vmin=0,
        yticklabels=anesthetic_names,
        xticklabels=region_abbr,
        cbar_kws={'label': 'Standardized Contribution'},
        ax=ax
    )

    for g, name in enumerate(anesthetic_names):
        for idx in unique_regions[name]:
            count = region_count[idx]
            if count == 1:
                marker = '★'
                color = 'black'
                unique_brain_regions[name].append(region_abbr[idx])
            else:
                marker = '●'
                color = 'green'
                if region_abbr[idx] not in shared_brain_regions:
                    shared_brain_regions.append(region_abbr[idx])

            ax.annotate(marker, xy=(idx + 0.5, g + 0.5),
                        ha='center', va='center',
                        color=color, fontsize=12,
                        weight='bold')

    ax.set_title('Brain Region Contributions in Anesthetic Groups\n'
                 '★: Unique to this anesthetic, ●: Shared by multiple anesthetics',
                 fontsize=16)
    plt.subplots_adjust(bottom=0.7)
    ax.set_xlabel('Brain Region (Abbreviation)', fontsize=14)
    ax.set_ylabel('Anesthetic', fontsize=14)
    plt.xticks(rotation=90, ha='center')
    plt.savefig(save_name + '.pdf', format="pdf")
    plt.savefig(save_name + '.png')
    plt.show()

    return unique_brain_regions, shared_brain_regions


def compute_all_region_p_values(contrib_matrix, anesthetic_names, all_anesthetic_data, alpha=0.05):
    n_regions = contrib_matrix.shape[1]
    p_values = []

    for region_idx in range(n_regions):
        values, labels = [], []
        for g, gname in enumerate(anesthetic_names):
            current_region_data = all_anesthetic_data[gname][:, region_idx]
            if current_region_data.size == 0:
                continue
            values.extend(current_region_data.flatten())
            labels.extend([g] * current_region_data.shape[0])

        if len(labels) < 2:
            p_values.append(np.nan)
            continue

        labels_np = np.array(labels)
        values_np = np.array(values)
        valid_groups = np.unique(labels_np)
        group_data_list = [values_np[labels_np == g] for g in valid_groups]
        group_data_list = [data for data in group_data_list if data.size > 0]

        if len(group_data_list) < 2:
            p_values.append(np.nan)
        else:
            _, p_val = kruskal(*group_data_list)
            p_values.append(p_val)

    return pd.Series(p_values, index=range(n_regions))

if __name__ == '__main__':
    topN = 10
    # all
    # brain_name_path = 'data/brain_regions.xlsx'
    # brain_data_path = 'data/brain1.npy'
    # contribution_data_path = "./results/five_anesthetics_contribution/femaleAndMale/group_imp.xlsx"
    # save_name = f'./results/five_anesthetics_contribution_anesthesia_differences-top{topN}/femaleAndMale'

    # female
    # brain_name_path = 'data/brain_regions.xlsx'
    # brain_data_path = 'data/brain1.npy'
    # contribution_data_path = './results/five_anesthetics_contribution/female/group_imp.xlsx'
    # save_name = f'./results/five_anesthetics_contribution_anesthesia_differences-top{topN}/female'

    # male
    brain_name_path = 'data/brain_regions.xlsx'
    brain_data_path = 'data/brain1.npy'
    contribution_data_path = r'./results/five_anesthetics_contribution/male/group_imp.xlsx'
    save_name = fr'./results/five_anesthetics_contribution_anesthesia_differences-top{topN}/male'
    os.makedirs(f'./results/five_anesthetics_contribution_anesthesia_differences-top{topN}', exist_ok=True)

    label_def = pd.read_excel(brain_name_path)
    label_names = label_def['Brain_Region'].tolist()

    df_contribution = pd.read_excel(contribution_data_path, header=0, index_col=0)
    df_contribution = df_contribution.iloc[:-1, :]
    anesthetic_names = df_contribution.index.tolist()

    blood_all_data = np.load(brain_data_path, allow_pickle=True).item()

    # all
    # all_anesthetic_data = {
    #     'Dex': filter_merge_and_remap_labels(blood_all_data, target_labels=[['dex_conscious_female', 'dex_conscious_male', 'dex_anesthesia_female', 'dex_anesthesia_male']])['matrix'],
    #     'ISO': filter_merge_and_remap_labels(blood_all_data, target_labels=[['iso_conscious_female', 'iso_conscious_male', 'iso_anesthesia_female', 'iso_anesthesia_male']])['matrix'],
    #     'Ketamine': filter_merge_and_remap_labels(blood_all_data, target_labels=[['ketamine_conscious_female', 'ketamine_conscious_male', 'ketamine_anesthesia_female', 'ketamine_anesthesia_male']])['matrix'],
    #     'N2O': filter_merge_and_remap_labels(blood_all_data, target_labels=[['N2O_conscious_female', 'N2O_anesthesia_male', 'N2O_conscious_female', 'N2O_anesthesia_male']])['matrix'],
    #     'Propofol': filter_merge_and_remap_labels(blood_all_data, target_labels=[['propofol_conscious_female', 'propofol_conscious_male', 'propofol_anesthesia_female', 'propofol_anesthesia_male']])['matrix']
    # }

    # female
    # all_anesthetic_data = {
    #     'Dex': filter_merge_and_remap_labels(blood_all_data, target_labels=[['dex_conscious_female', 'dex_anesthesia_female']])['matrix'],
    #     'ISO': filter_merge_and_remap_labels(blood_all_data, target_labels=[['iso_conscious_female', 'iso_anesthesia_female']])['matrix'],
    #     'Ketamine': filter_merge_and_remap_labels(blood_all_data, target_labels=[['ketamine_conscious_female', 'ketamine_anesthesia_female']])['matrix'],
    #     'N2O': filter_merge_and_remap_labels(blood_all_data, target_labels=[['N2O_conscious_female', 'N2O_anesthesia_female']])['matrix'],
    #     'Propofol': filter_merge_and_remap_labels(blood_all_data, target_labels=[['propofol_conscious_female', 'propofol_anesthesia_female']])['matrix']
    # }

    # male
    all_anesthetic_data = {
        'Dex': filter_merge_and_remap_labels(blood_all_data, target_labels=[['dex_conscious_male', 'dex_anesthesia_male']])['matrix'],
        'ISO': filter_merge_and_remap_labels(blood_all_data, target_labels=[['iso_conscious_male', 'iso_anesthesia_male']])['matrix'],
        'Ketamine': filter_merge_and_remap_labels(blood_all_data, target_labels=[['ketamine_conscious_male', 'ketamine_anesthesia_male']])['matrix'],
        'N2O': filter_merge_and_remap_labels(blood_all_data, target_labels=[['N2O_conscious_male', 'N2O_anesthesia_male']])['matrix'],
        'Propofol': filter_merge_and_remap_labels(blood_all_data, target_labels=[['propofol_conscious_male', 'propofol_anesthesia_male']])['matrix']
    }

    unique_regions, _ = find_unique_regions(df_contribution, anesthetic_names, all_anesthetic_data,
                                            threshold_mode=f'top_{topN}')
    unique_brain_regions, shared_brain_regions = plot_contrib_heatmap(df_contribution, unique_regions, label_names,
                                                                      anesthetic_names, save_name=save_name)

    with pd.ExcelWriter(save_name + '_regions.xlsx') as writer:

        max_len = max(len(regions) for regions in unique_brain_regions.values())
        unique_df = pd.DataFrame({
            'Anesthetic': [],
            'Unique_Regions': []
        })

        for name, regions in unique_brain_regions.items():
            temp_df = pd.DataFrame({
                'Anesthetic': [name] * len(regions),
                'Unique_Regions': regions
            })
            unique_df = pd.concat([unique_df, temp_df], ignore_index=True)

        unique_df.to_excel(writer, sheet_name='Unique_Regions', index=False)

        shared_df = pd.DataFrame({
            'Shared_Regions': shared_brain_regions
        })
        shared_df.to_excel(writer, sheet_name='Shared_Regions', index=False)

