import os
import numpy as np
import sys


from utils_refactored import (
    analyze_cross_anesthetic_specificity,
    analyze_within_receptor_comparison,
    load_brain_regions_mapping,
    load_raw_data_for_anesthetic,
    load_importance_matrix
)

# Global random seed
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

def main():
    """
    Main function for specificity analysis across five anesthetics.

    Functions:
    1. Load raw data and feature-importance matrices for each anesthetic.
    2. Perform cross-anesthetic specificity analysis.
    3. Perform within-receptor binary comparison analysis.
    """
    print("\n" + "=" * 70)
    print("🔬 Specificity Analysis for Five Anesthetics")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Configure paths
    # ------------------------------------------------------------------
    brain_regions_mapping_file = r'Allen_Brain_Regions(1).xlsx'
    base_input_dir = r'data'
    base_output_dir = r'output/five_anesthetics_specificity_rf'

    # Anesthetic groups (5 anesthetics + all)
    ANESTHETIC_GROUPS = ['dex', 'iso', 'ketamine', 'n2o', 'propofol', 'all']
    SPECIFIC_GROUPS = ['dex', 'iso', 'ketamine', 'n2o', 'propofol']

    # Threshold modes for specificity analysis
    THRESHOLD_MODES = ['top_10', 'top_50']

    # ------------------------------------------------------------------
    # Load brain-region mapping
    # ------------------------------------------------------------------
    try:
        brain_regions, brain_regions_abbr = load_brain_regions_mapping(brain_regions_mapping_file)
        print(f"✅ Brain-region mapping loaded: {len(brain_regions)} full names, {len(brain_regions_abbr)} abbreviations\n")
    except FileNotFoundError as e:
        print(f"❌ Error: Brain-region mapping file not found: {e}")
        return

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("📂 Loading raw data and importance matrices...\n")

    raw_data_store = {}
    importance_matrices_dict = {}

    for anes_type in SPECIFIC_GROUPS:
        print(f"--- Processing {anes_type.upper()} ---")

        # Load raw data
        raw_data, _, pos_count, neg_count = load_raw_data_for_anesthetic(anes_type, base_input_dir)
        if raw_data is not None:
            raw_data_store[anes_type] = raw_data

        # Load feature-importance matrix
        matrix_file = os.path.join(
            base_output_dir,
            "five_anesthetics_contribution_rf",
            anes_type,
            f"{anes_type}_rf_feature_importance_matrix.csv"
        )
        importance_matrix = load_importance_matrix(matrix_file)
        if importance_matrix is not None:
            importance_matrices_dict[anes_type] = importance_matrix

    # ------------------------------------------------------------------
    # Data integrity check
    # ------------------------------------------------------------------
    print("=" * 70)
    print("🔍 Data integrity check...")
    print("=" * 70)
    print(f"Anesthetics in raw_data_store: {list(raw_data_store.keys())}")
    print(f"Number of entries in raw_data_store: {len(raw_data_store)}")
    if raw_data_store:
        for k, v in raw_data_store.items():
            print(f"  - {k}: shape = {v.shape}")

    print(f"\nAnesthetics in importance_matrices_dict: {list(importance_matrices_dict.keys())}")
    print(f"Number of entries in importance_matrices_dict: {len(importance_matrices_dict)}")
    if importance_matrices_dict:
        for k, v in importance_matrices_dict.items():
            print(f"  - {k}: shape = {v.shape}")

    print("\n" + "=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Specificity analysis
    # ------------------------------------------------------------------
    if not raw_data_store or not importance_matrices_dict:
        print("❌ Incomplete data; specificity analysis cannot be performed.")
        if not raw_data_store:
            print("   ❌ raw_data_store is empty")
        if not importance_matrices_dict:
            print("   ❌ importance_matrices_dict is empty")
        return

    print(f"✅ Data ready. Results will be generated for threshold modes: {THRESHOLD_MODES}")
    print("=" * 70 + "\n")

    # Loop over each threshold mode
    for threshold_mode in THRESHOLD_MODES:
        print("\n" + "=" * 70)
        print(f"🔬 Processing threshold_mode = {threshold_mode}")
        print("=" * 70)

        # Create separate output directories for each threshold
        cross_anesthetic_specificity_output_dir = os.path.join(
            base_output_dir,
            "Functional Connectivity - Specific High-Contribution Features for 5 Anesthetics Separately",
            "Sex-Combined"
        )
        within_receptor_comparison_output_dir = os.path.join(
            base_output_dir,
            "Functional Connectivity - Specificity Between Two Anesthetics Within the Same Receptor - FC Data",
            threshold_mode
        )
        os.makedirs(cross_anesthetic_specificity_output_dir, exist_ok=True)
        os.makedirs(within_receptor_comparison_output_dir, exist_ok=True)

        print(f"\n📁 Output directory: {cross_anesthetic_specificity_output_dir}")
        print("🔬 Running cross-anesthetic specificity analysis...")

        # 1) Cross-anesthetic specificity analysis
        analyze_cross_anesthetic_specificity(
            raw_data_store=raw_data_store,
            importance_matrices_dict=importance_matrices_dict,
            brain_regions=brain_regions,
            base_output_dir=cross_anesthetic_specificity_output_dir,
            alpha=0.05,
            threshold_mode=threshold_mode,
        )

        print(f"\n🔬 Running within-receptor binary comparison analysis (threshold={threshold_mode})...")

        # 2) Within-receptor binary comparison analysis
        analyze_within_receptor_comparison(
            raw_data_store=raw_data_store,
            importance_matrices_dict=importance_matrices_dict,
            brain_regions=brain_regions,
            base_output_dir=within_receptor_comparison_output_dir,
            alpha=0.05,
            threshold_mode=threshold_mode,
        )

        print(f"\n✅ threshold_mode = {threshold_mode} completed")
        print("=" * 70)

    print(f"\n🎉 All threshold modes completed! Total modes: {len(THRESHOLD_MODES)}")
    print("\n🎉 Specificity analysis workflow completed.\n")

if __name__ == "__main__":
    main()