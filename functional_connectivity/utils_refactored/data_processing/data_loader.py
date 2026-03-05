"""
Data loading module
Includes functions for loading raw data, brain region mappings,
and importance matrices.
"""

import os
import re
import numpy as np
import pandas as pd


def read_excel_data(file_path, label):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")

    # Prefer loading .npy (10-100x faster) ⚡
    npy_path = file_path.replace(".xlsx", ".npy")

    if os.path.exists(npy_path):
        # Load with NumPy (fast)
        connectivity_matrix = np.load(npy_path)
    else:
        # Fallback to Excel (slow but compatible)
        df = pd.read_excel(file_path, header=None, index_col=None)
        connectivity_matrix = df.values

    connectivity_matrix = np.where(connectivity_matrix == 0, 1e-8, connectivity_matrix)
    n = connectivity_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    upper_tri_data = connectivity_matrix[upper_tri_indices]
    return upper_tri_data, label, n


def load_brain_regions_mapping(mapping_file):
    df = pd.read_excel(mapping_file, header=0)

    # Load full names (remove parentheses)
    name_col = "Name" if "Name" in df.columns else "Allen Name"
    df[name_col] = df[name_col].apply(lambda x: re.sub(r"\s*\([^)]*\)", "", str(x)))
    mapping_full = df[name_col].reset_index().set_index("index")[name_col].to_dict()

    # Load abbreviations
    abbr_col = "Allen Abbreviation"
    if abbr_col in df.columns:
        mapping_abbr = df[abbr_col].reset_index().set_index("index")[abbr_col].to_dict()
    else:
        # If no abbreviation column, use the first 5 chars of the full name as a fallback
        mapping_abbr = {k: str(v)[:5] for k, v in mapping_full.items()}

    return mapping_full, mapping_abbr


def load_raw_data_for_anesthetic(anes_type, base_input_dir):
    data_list = []
    label_list = []

    # Load positive samples (Label=1, anesthetic)
    pos_folder = os.path.join(base_input_dir, f"anesthetic_{anes_type}", "all")
    if os.path.exists(pos_folder):
        files = sorted([file for file in os.listdir(pos_folder) if file.endswith(".xlsx")])
        for file in files:
            file_path = os.path.join(pos_folder, file)
            try:
                features, label, _ = read_excel_data(file_path, 1)
                data_list.append(features)
                label_list.append(label)
            except Exception:
                continue
    pos_count = len(data_list)

    # Load negative samples (Label=0, non-anesthetic)
    neg_folder = os.path.join(base_input_dir, f"non_anesthetic_{anes_type}", "all")
    if os.path.exists(neg_folder):
        files = sorted([file for file in os.listdir(neg_folder) if file.endswith(".xlsx")])
        for file in files:
            file_path = os.path.join(neg_folder, file)
            try:
                features, label, _ = read_excel_data(file_path, 0)
                data_list.append(features)
                label_list.append(label)
            except Exception:
                continue
    neg_count = len(data_list) - pos_count

    if data_list:
        print(
            f"✅ Loaded raw data for {anes_type}: {len(data_list)} samples "
            f"(positive={pos_count}, negative={neg_count})"
        )
        return np.array(data_list), np.array(label_list), pos_count, neg_count
    else:
        print(f"⚠️ No valid data found for {anes_type}")
        return None, None, 0, 0


def load_importance_matrix(matrix_file):
    if os.path.exists(matrix_file):
        importance_matrix = pd.read_csv(matrix_file, header=None).values
        print(f"✅ Loaded importance matrix: {matrix_file}, shape={importance_matrix.shape}")
        return importance_matrix
    else:
        print(f"⚠️ Importance matrix file does not exist: {matrix_file}")
        return None


def load_gender_data_for_anesthetic(anesthetic_type, gender, base_input_dir, base_output_dir):
    from tqdm import tqdm

    print(f"\nProcessing {gender.upper()} - {anesthetic_type.upper()}...")

    # 1. Load importance matrix
    matrix_path = os.path.join(
        base_output_dir,
        "HighContributionByDrugSeparatedByGender",
        gender,
        anesthetic_type,
        f"{anesthetic_type}_rf_feature_importance_matrix.csv",
    )
    if not os.path.exists(matrix_path):
        print(f"⚠️ Importance matrix not found: {matrix_path}")
        return None, None, 0, 0

    importance_matrix = pd.read_csv(matrix_path, header=None).values
    print(f"✅ Loaded importance matrix: {importance_matrix.shape}")

    # 2. Load raw data
    data_list = []

    # Load positive samples
    pos_data_folder = os.path.join(base_input_dir, f"anesthetic_{anesthetic_type}", gender)
    pos_count = 0
    if os.path.exists(pos_data_folder):
        files = sorted([f for f in os.listdir(pos_data_folder) if f.endswith(".xlsx")])
        for file in tqdm(files, desc="Loading positive samples", leave=False):
            file_path = os.path.join(pos_data_folder, file)
            try:
                features, _, _ = read_excel_data(file_path, 1)
                data_list.append(features)
            except Exception:
                continue
        pos_count = len(data_list)

    # Load negative samples
    neg_data_folder = os.path.join(base_input_dir, f"non_anesthetic_{anesthetic_type}", gender)
    neg_count = 0
    if os.path.exists(neg_data_folder):
        files = sorted([f for f in os.listdir(neg_data_folder) if f.endswith(".xlsx")])
        for file in tqdm(files, desc="Loading negative samples", leave=False):
            file_path = os.path.join(neg_data_folder, file)
            try:
                features, _, _ = read_excel_data(file_path, 0)
                data_list.append(features)
            except Exception:
                continue
        neg_count = len(data_list) - pos_count

    if not data_list:
        print("⚠️ Failed to load any data")
        return None, None, 0, 0

    raw_data = np.array(data_list)
    print(f"✅ Loaded raw data: {len(data_list)} samples (positive={pos_count}, negative={neg_count})")

    return raw_data, importance_matrix, pos_count, neg_count