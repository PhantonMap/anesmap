
import os
import warnings
import numpy as np
from utils_refactored import (
    load_brain_regions_mapping,
    analyze_cross_anesthetic_specificity,
    analyze_receptor_specificity,
    analyze_within_receptor_comparison,
    analyze_gender_specificity,
    load_gender_data_for_anesthetic
)

# Suppress all warnings
warnings.filterwarnings("ignore")

# Global random seed
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


def main():
    """
    Main function: run the complete gender-specific analysis, including:
    1. Between-gender specificity analysis: for each anesthetic, compare Male vs Female
    2. Within-gender specificity analysis: within each gender, compare specificity across the 5 anesthetics
    """
    print("\n" + "="*80)
    print("🔬🔬🔬 Full Gender Specificity Analysis 🔬🔬🔬")
    print("🔹 Between-gender specificity: Male vs Female (for each anesthetic)")
    print("🔹 Within-gender specificity: comparison across 5 anesthetics (within each gender)")
    print("="*80 + "\n")

    # ------------------------------------------------------------------
    # Configure paths
    # ------------------------------------------------------------------
    brain_regions_mapping_file = r'Allen_Brain_Regions(1).xlsx'
    base_input_dir = r'data'
    base_output_dir = r'output/five_anesthetics_sex_separated_specificity_rf'

    SPECIFIC_GROUPS = ['dex', 'iso', 'ketamine', 'n2o', 'propofol']
    GENDERS = ['male', 'female']
    THRESHOLD_MODES = ['top_10', 'top_50']

    # ------------------------------------------------------------------
    # Load brain-region mapping
    # ------------------------------------------------------------------
    print("📂 Loading brain-region mapping...")
    try:
        brain_regions, brain_regions_abbr = load_brain_regions_mapping(brain_regions_mapping_file)
        print(f"✅ Brain-region mapping loaded: {len(brain_regions)} full names, {len(brain_regions_abbr)} abbreviations\n")
    except FileNotFoundError as e:
        print(f"❌ Error: brain-region mapping file not found: {e}")
        return

    # ==================================================================
    # Part 1: Between-gender specificity analysis (Male vs Female for each anesthetic)
    # ==================================================================
    print("\n" + "="*80)
    print("📊 Part 1: Between-gender specificity analysis (Male vs Female)")
    print("="*80)
    print("For each anesthetic, analyze specificity differences between Male and Female\n")

    for threshold_mode in THRESHOLD_MODES:
        print("\n" + "="*80)
        print(f"🔬 Starting between-gender specificity analysis with threshold_mode = {threshold_mode}")
        print("="*80 + "\n")

        gender_specificity_output_dir = os.path.join(base_output_dir, "five_anesthetics_sex_separated_high_contribution_rf")
        os.makedirs(gender_specificity_output_dir, exist_ok=True)

        # This section compares genders within each anesthetic (not cross-anesthetic),
        # so it naturally includes "all".
        for anes_type in ['dex', 'iso', 'ketamine', 'n2o', 'propofol', 'all']:
            print(f"\n--- {anes_type.upper()} (threshold={threshold_mode}) ---")

            # Run between-gender specificity analysis
            analyze_gender_specificity(
                anesthetic_type=anes_type,
                base_input_dir=base_input_dir,
                brain_regions=brain_regions,
                brain_regions_abbr=brain_regions_abbr,
                base_output_dir=base_output_dir,
                alpha=0.05,
                threshold_mode=threshold_mode
            )

            print(f"✅ {anes_type.upper()} completed (threshold={threshold_mode})")

        print(f"\n✅ Between-gender specificity analysis completed for all anesthetics (threshold_mode = {threshold_mode})!")
        print("="*80)

    print("\n" + "="*80)
    print("🎉 Between-gender specificity analysis completed for all anesthetics!")
    print("="*80 + "\n")

    # ==================================================================
    # Part 2: Within-gender specificity analysis (comparison across 5 anesthetics within each gender)
    # ==================================================================
    print("\n" + "="*80)
    print("📊 Part 2: Within-gender specificity analysis (comparison across 5 anesthetics)")
    print("="*80)
    print("Within each gender, analyze specificity across the 5 anesthetics\n")

    # ------------------------------------------------------------------
    # Loop over genders
    # ------------------------------------------------------------------
    for gender in GENDERS:
        print("\n" + "="*80)
        print(f"🔬🔬🔬 Processing {gender.upper()} group specificity analysis 🔬🔬🔬")
        print("="*80 + "\n")

        # Load all anesthetic data for the current gender
        raw_data_store = {}
        importance_matrices_dict = {}

        print(f"📂 Loading raw data and importance matrices for {gender.upper()}...\n")

        for anes_type in SPECIFIC_GROUPS:
            raw_data, importance_matrix, pos_count, neg_count = load_gender_data_for_anesthetic(
                anes_type, gender, base_input_dir, base_output_dir
            )

            if raw_data is not None:
                raw_data_store[anes_type] = raw_data
            if importance_matrix is not None:
                importance_matrices_dict[anes_type] = importance_matrix

        # ------------------------------------------------------------------
        # Data integrity check
        # ------------------------------------------------------------------
        print("\n" + "="*70)
        print(f"🔍 Data integrity check ({gender.upper()})...")
        print("="*70)
        print(f"Anesthetics in raw_data_store: {list(raw_data_store.keys())}")
        print(f"raw_data_store count: {len(raw_data_store)}")
        if raw_data_store:
            for k, v in raw_data_store.items():
                print(f"  - {k}: shape = {v.shape}")

        print(f"\nAnesthetics in importance_matrices_dict: {list(importance_matrices_dict.keys())}")
        print(f"importance_matrices_dict count: {len(importance_matrices_dict)}")
        if importance_matrices_dict:
            for k, v in importance_matrices_dict.items():
                print(f"  - {k}: shape = {v.shape}")

        print("\n" + "="*70 + "\n")

        # ------------------------------------------------------------------
        # Cross-anesthetic specificity analysis (within gender)
        # ------------------------------------------------------------------
        if raw_data_store and importance_matrices_dict:
            # Only analyze anesthetics that have BOTH raw data and an importance matrix
            common_keys = sorted(set(raw_data_store.keys()) & set(importance_matrices_dict.keys()))
            missing_raw = sorted(set(importance_matrices_dict.keys()) - set(raw_data_store.keys()))
            missing_imp = sorted(set(raw_data_store.keys()) - set(importance_matrices_dict.keys()))

            if missing_raw:
                print(f"⚠️ Present in importance_matrices_dict but missing in raw_data_store: {missing_raw} (will not analyze these groups)")
            if missing_imp:
                print(f"⚠️ Present in raw_data_store but missing in importance_matrices_dict: {missing_imp} (will not analyze these groups)")

            if common_keys:
                print(f"✅ Data preparation completed ({gender.upper()}). Results will be generated for these threshold modes: {THRESHOLD_MODES}")
                print(f"   Total {len(common_keys)} anesthetics: {common_keys}")
                print("="*70 + "\n")

                # Loop over threshold modes
                for threshold_mode in THRESHOLD_MODES:
                    print("\n" + "="*70)
                    print(f"🔬 Processing threshold_mode = {threshold_mode} ({gender.upper()})")
                    print("="*70)

                    # Create an independent output directory for each threshold
                    threshold_output_dir = os.path.join(
                        base_output_dir,
                        'Functional Connectivity – High-Contribution Connections Specific to Each of the Five Anesthetics',
                        gender,
                    )
                    os.makedirs(threshold_output_dir, exist_ok=True)

                    print(f"\n📁 Output directory: {threshold_output_dir}")
                    print(f"🔬 Starting cross-anesthetic specificity analysis (threshold={threshold_mode})...")

                    # 1. Cross-anesthetic specificity analysis
                    analyze_cross_anesthetic_specificity(
                        raw_data_store=raw_data_store,
                        importance_matrices_dict=importance_matrices_dict,
                        brain_regions=brain_regions,
                        base_output_dir=threshold_output_dir,
                        alpha=0.05,
                        threshold_mode=threshold_mode,
                    )


                    print(f"\n✅ threshold_mode = {threshold_mode} completed ({gender.upper()})")
                    print("="*70)

                print(f"\n🎉 All threshold modes completed! Total {len(THRESHOLD_MODES)} modes ({gender.upper()})")
            else:
                print(f"⚠️ No anesthetic has BOTH raw data and importance matrix. Skipping ({gender.upper()}).")
        else:
            print(f"⚠️ Required data for cross-anesthetic specificity analysis is incomplete. Skipping ({gender.upper()}).")
            if not raw_data_store:
                print("   ❌ raw_data_store is empty")
            if not importance_matrices_dict:
                print("   ❌ importance_matrices_dict is empty")

        print(f"\n🎉 All workflows finished for {gender.upper()} group (including cross-anesthetic specificity analysis + receptor grouping analysis + within-receptor binary comparisons).")
        print("="*80 + "\n")

    # ==================================================================
    # Done
    # ==================================================================
    print("\n" + "="*80)
    print("🎉🎉🎉 Full gender-specific analysis completed! 🎉🎉🎉")


if __name__ == "__main__":
    main()