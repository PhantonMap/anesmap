"""
Receptor analysis module
Functions for contribution analysis and statistical tests grouped by receptor class.
"""

import os
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn


def analyze_by_receptor(region_importances_dict, x_axis_labels, base_output_dir, top_k=20, alpha=0.05):
    """
    Group brain-region contributions by receptor class (NMDA / GABA / a2) and perform:
    1) Contribution ranking (mean contribution within each receptor group)
    2) Kruskal–Wallis test
    3) Dunn's post-hoc test
       (Only truly meaningful when number of groups >= 3 and KW is significant;
        here we still output the results for reference.)

    Args:
        region_importances_dict: dict. Keys are anesthetic names, values are region-importance vectors.
        x_axis_labels: list of region labels.
        base_output_dir: root directory for outputs.
        top_k: show top-K regions (kept for compatibility; visualization removed).
        alpha: significance level.
    """
    # Mapping: receptor -> anesthetics under this receptor (short names used in anes_type)
    receptor_groups = {
        "NMDA": ["ketamine", "n2o"],
        "GABA": ["iso", "propofol"],
        "a2":   ["dex"],  # Only one anesthetic; no meaningful between-group test
    }

    for receptor, drug_list in receptor_groups.items():
        # Filter out anesthetics that were not successfully trained/available in this run
        available_drugs = [d for d in drug_list if d in region_importances_dict]
        if len(available_drugs) == 0:
            print(f"⚠️ No available drugs for receptor {receptor}; skipping.")
            continue

        # Stack region-importance vectors into a matrix: shape = (n_drugs, n_regions)
        import numpy as np
        contrib_matrix = np.stack([region_importances_dict[d] for d in available_drugs], axis=0)
        n_drugs, n_regions = contrib_matrix.shape

        receptor_dir = os.path.join(base_output_dir, f"receptor_{receptor}")
        os.makedirs(receptor_dir, exist_ok=True)

        drug_labels = [d.upper() for d in available_drugs]

        if len(available_drugs) >= 2:
            group_data_list = [region_importances_dict[d] for d in available_drugs]
            kw_stat, kw_p = kruskal(*group_data_list)
            print(f"Receptor {receptor} Kruskal-Wallis: stat={kw_stat:.4f}, p={kw_p:.4e}")

            with open(os.path.join(receptor_dir, f"{receptor}_kruskal_result.txt"), "w", encoding="utf-8") as f:
                f.write(f"Receptor: {receptor}\n")
                f.write(f"Drugs: {', '.join(available_drugs)}\n")
                f.write(f"Kruskal-Wallis H = {kw_stat:.6f}, p = {kw_p:.6e}\n")

            # ------------------------
            # 3) Dunn's post-hoc test
            # Pairwise comparisons of region-importance distributions between drugs within the receptor group.
            # ------------------------
            try:
                dunn_res = posthoc_dunn(group_data_list, p_adjust='bonferroni')
                dunn_res.index = drug_labels
                dunn_res.columns = drug_labels
                import pandas as pd
                dunn_res.to_excel(os.path.join(receptor_dir, f"{receptor}_dunn_posthoc.xlsx"))
            except Exception as e:
                print(f"⚠️ Dunn test failed for receptor {receptor}: {e}")
        else:
            print(f"Receptor {receptor} has only one drug ({available_drugs[0]}), "
                  f"so no Kruskal/Dunn between-group test is performed.")