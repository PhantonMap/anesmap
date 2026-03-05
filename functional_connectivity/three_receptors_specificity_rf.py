import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings


from utils_refactored import analyze_cross_anesthetic_specificity


warnings.filterwarnings("ignore")


def read_excel_data(file_path, label):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    # Prefer loading the .npy file first
    npy_path = file_path.replace('.xlsx', '.npy')
    
    if os.path.exists(npy_path):
        # Load with NumPy (10-100x faster) ⚡
        connectivity_matrix = np.load(npy_path)
    else:
        # Fallback to Excel (slower but compatible)
        df = pd.read_excel(file_path, header=None, index_col=None)
        connectivity_matrix = df.values
    
    connectivity_matrix = np.where(connectivity_matrix == 0, 1e-8, connectivity_matrix)
    n = connectivity_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    upper_tri_data = connectivity_matrix[upper_tri_indices]
    return upper_tri_data, label, n


def load_brain_regions_mapping(mapping_file):

    df = pd.read_excel(mapping_file, header=0)
    
    # Load full names (remove content inside parentheses)
    name_col = 'Name' if 'Name' in df.columns else 'Allen Name'
    df[name_col] = df[name_col].apply(lambda x: re.sub(r"\s*\([^)]*\)", "", str(x)))
    mapping_full = df[name_col].reset_index().set_index('index')[name_col].to_dict()
    
    # Load abbreviations
    abbr_col = 'Allen Abbreviation'
    if abbr_col in df.columns:
        mapping_abbr = df[abbr_col].reset_index().set_index('index')[abbr_col].to_dict()
    else:
        # If there is no abbreviation column, use the first 5 characters of the full name as a fallback
        mapping_abbr = {k: str(v)[:5] for k, v in mapping_full.items()}
    
    return mapping_full, mapping_abbr


def load_receptor_raw_data(base_input_dir, anesthetic_list):

    all_data_list = []
    
    label_folder = "all"  # Use all-sex data
    
    # Load positive samples (Label=1, anesthetic)
    for anes_type in anesthetic_list:
        pos_folder = os.path.join(base_input_dir, f"anesthetic_{anes_type}", label_folder)
        if os.path.exists(pos_folder):
            files = sorted([f for f in os.listdir(pos_folder) if f.endswith('.xlsx')])
            desc = f"  Loading {anes_type} positive samples"
            for file in tqdm(files, desc=desc, unit="file", leave=False):
                file_path = os.path.join(pos_folder, file)
                try:
                    features, _, _ = read_excel_data(file_path, 1)
                    all_data_list.append(features)
                except Exception:
                    continue
        else:
            print(f"  ⚠️ Folder does not exist: {pos_folder}")
    
    pos_count = len(all_data_list)
    
    # Load negative samples (Label=0, non-anesthetic)
    for anes_type in anesthetic_list:
        neg_folder = os.path.join(base_input_dir, f"non_anesthetic_{anes_type}", label_folder)
        if os.path.exists(neg_folder):
            files = sorted([f for f in os.listdir(neg_folder) if f.endswith('.xlsx')])
            desc = f"  Loading {anes_type} negative samples"
            for file in tqdm(files, desc=desc, unit="file", leave=False):
                file_path = os.path.join(neg_folder, file)
                try:
                    features, _, _ = read_excel_data(file_path, 0)
                    all_data_list.append(features)
                except Exception:
                    continue
        else:
            print(f"  ⚠️ Folder does not exist: {neg_folder}")
    
    neg_count = len(all_data_list) - pos_count
    
    if all_data_list:
        print(f"  ✅ Loaded data: {len(all_data_list)} samples (positive={pos_count}, negative={neg_count})")
        return np.array(all_data_list)
    else:
        return None


# ------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------

def main():
    print("\n" + "="*70)
    print("🔬 Inter-receptor specificity analysis (standalone script)")
    print("="*70)
    
    # Brain region mapping path
    brain_regions_mapping_file = r'Allen_Brain_Regions(1).xlsx'
    base_input_dir = r'data'
    base_output_dir = r'output/three_receptors_specificity_rf'

    try:
        brain_regions, brain_regions_abbr = load_brain_regions_mapping(brain_regions_mapping_file)
        print(f"✅ Loaded brain region mapping: {len(brain_regions)} full names, {len(brain_regions_abbr)} abbreviations")
    except FileNotFoundError as e:
        print(f"Error: Brain region mapping file not found: {e}")
        return

    # Define receptor groups
    receptor_groups = {
        'GABA': ['iso', 'propofol'],
        'NMDA': ['ketamine', 'n2o'],
        'α2': ['dex']
    }

    THRESHOLD_MODES = ['top_10', 'top_50']

    
    # Check whether the model output directory exists
    if not os.path.exists(base_output_dir):
        print(f"❌ Model output directory does not exist: {base_output_dir}")
        return
    
    # ------------------------------------------------------------------
    # Load the saved importance matrices and raw data
    # ------------------------------------------------------------------
    raw_data_store = {}  # Store raw data (without SMOTE)
    importance_matrices_dict = {}  # Store importance matrices
    
    print("\n" + "="*70)
    print("📂 Start loading importance matrices and raw data...")
    print("="*70)
    
    for receptor_name, anesthetic_list in receptor_groups.items():
        print(f"\n🔍 Processing receptor: {receptor_name} ({', '.join(anesthetic_list)})")
        
        
        matrix_file = os.path.join(
            base_output_dir,
            "three_receptors_top_contributing_brain_regions_rf",
            f"receptor_{receptor_name}",
            f"receptor_{receptor_name}_rf_feature_importance_matrix.csv"
        )
        
        # 1. Load importance matrix
        if os.path.exists(matrix_file):
            importance_matrix = pd.read_csv(matrix_file, header=None).values
            importance_matrices_dict[receptor_name] = importance_matrix
            print(f"  ✅ Loaded importance matrix: {importance_matrix.shape}")
        else:
            print(f"  ❌ Importance matrix file does not exist: {matrix_file}")
            print("     Please run the model training script to generate this file")
            continue
        
        # 2. Load raw data
        print("  📂 Loading raw data...")
        raw_data = load_receptor_raw_data(base_input_dir, anesthetic_list)
        if raw_data is not None:
            raw_data_store[receptor_name] = raw_data
        else:
            print(f"  ⚠️ Failed to load raw data for {receptor_name}")
    
    # ------------------------------------------------------------------
    # Data integrity check
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("🔍 Data integrity check...")
    print("="*70)
    print(f"Receptors with loaded importance matrices: {list(importance_matrices_dict.keys())}")
    print(f"Receptors with loaded raw data: {list(raw_data_store.keys())}")
    
    if importance_matrices_dict:
        for k, v in importance_matrices_dict.items():
            print(f"  - {k} importance matrix: shape = {v.shape}")
    
    if raw_data_store:
        for k, v in raw_data_store.items():
            print(f"  - {k} raw data: shape = {v.shape}")
    
    print("="*70 + "\n")

    
    if raw_data_store and importance_matrices_dict and len(raw_data_store) >= 2:
        print(f"✅ Data preparation complete. Results will be generated for threshold modes: {THRESHOLD_MODES}")
        print("="*70 + "\n")
        
        # Loop over each threshold mode
        for threshold_mode in THRESHOLD_MODES:
            print("\n" + "="*70)
            print(f"🔬 Start processing threshold_mode = {threshold_mode} (cross-receptor specificity analysis)")
            print("="*70)
            
            thresholdmode = threshold_mode.replace('_', '')
            # Create an independent output directory for each threshold
            specificity_output_dir = os.path.join(
                base_output_dir,
                f"Functional Connectivity - High-Contribution Connections Specific to Each of the Three Receptors",
                f"{thresholdmode}"
            )
            os.makedirs(specificity_output_dir, exist_ok=True)
            
            print(f"\n📁 Output directory: {specificity_output_dir}")
            print("🔬 Starting cross-receptor specificity analysis...")
            
            analyze_cross_anesthetic_specificity(
                raw_data_store=raw_data_store,
                importance_matrices_dict=importance_matrices_dict,
                brain_regions=brain_regions,
                base_output_dir=specificity_output_dir,
                alpha=0.05,
                threshold_mode=threshold_mode,
            )
            
            print(f"\n✅ threshold_mode = {threshold_mode} cross-receptor specificity analysis completed")
            print(f"   Results saved to: {specificity_output_dir}")
            print("="*70)
        
        print(f"\n🎉 All threshold modes completed! Total: {len(THRESHOLD_MODES)} modes")
    else:
        print("⚠️ Required data for cross-receptor specificity analysis is incomplete. Cannot proceed.")
       