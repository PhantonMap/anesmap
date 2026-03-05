import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import warnings

# ------------------------------------------------------------------
# 1. Import standard utility libraries
# ------------------------------------------------------------------
from metrics_tool import MetricsLogger, plot_all_curves
from utils_refactored import plot_importance_heatmap

# Disable all warnings
warnings.filterwarnings("ignore")

# Global random seed
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# ------------------------------------------------------------------
# Data loading and helper functions
# ------------------------------------------------------------------
def read_excel_data(file_path, label):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")

    # Prefer loading the .npy file
    npy_path = file_path.replace('.xlsx', '.npy')

    if os.path.exists(npy_path):
        # Load with NumPy (10-100x faster) ⚡
        connectivity_matrix = np.load(npy_path)
    else:
        # Fall back to Excel (slow but compatible)
        df = pd.read_excel(file_path, header=None, index_col=None)
        connectivity_matrix = df.values

    connectivity_matrix = np.where(connectivity_matrix == 0, 1e-8, connectivity_matrix)
    n = connectivity_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    upper_tri_data = connectivity_matrix[upper_tri_indices]
    return upper_tri_data, label, n


def load_brain_regions_mapping(mapping_file):
    """
    Load the mapping between full names and abbreviations from the Allen brain region file.

    Returns:
        brain_regions_full: dict {index: full_name}
        brain_regions_abbr: dict {index: abbreviation}
    """
    df = pd.read_excel(mapping_file, header=0)

    # Load full names (remove parentheses)
    name_col = 'Name' if 'Name' in df.columns else 'Allen Name'
    df[name_col] = df[name_col].apply(lambda x: re.sub(r"\s*\([^)]*\)", "", str(x)))
    mapping_full = df[name_col].reset_index().set_index('index')[name_col].to_dict()

    # Load abbreviations
    abbr_col = 'Allen Abbreviation'
    if abbr_col in df.columns:
        mapping_abbr = df[abbr_col].reset_index().set_index('index')[abbr_col].to_dict()
    else:
        # If abbreviation column is missing, use the first 5 characters of full name as fallback
        mapping_abbr = {k: str(v)[:5] for k, v in mapping_full.items()}

    return mapping_full, mapping_abbr


def load_receptor_data(base_input_dir, anesthetic_list, label):
    """
    Load data for a given anesthetic list (used for receptor grouping).

    Args:
        base_input_dir: base data directory
        anesthetic_list: list of anesthetics, e.g. ['iso', 'propofol']
        label: data label, 1=anesthetic, 0=non-anesthetic

    Returns:
        data_list: list of feature vectors
        labels_list: list of labels
        matrix_sizes: list of matrix sizes
    """
    data_list = []
    labels_list = []
    matrix_sizes = []

    label_folder = "all"  # Use all-sex combined data

    for anes_type in anesthetic_list:
        if label == 1:
            folder_path = os.path.join(base_input_dir, f"anesthetic_{anes_type}", label_folder)
        else:
            folder_path = os.path.join(base_input_dir, f"non_anesthetic_{anes_type}", label_folder)

        if os.path.exists(folder_path):
            files = sorted([f for f in os.listdir(folder_path) if f.endswith('.xlsx')])
            desc = f"Loading {anes_type} (label {label})"
            for file in tqdm(files, desc=desc, unit="file"):
                file_path = os.path.join(folder_path, file)
                try:
                    features, _, n = read_excel_data(file_path, label)
                    data_list.append(features)
                    labels_list.append(label)
                    matrix_sizes.append(n)
                except Exception:
                    continue
        else:
            print(f"⚠️ Folder does not exist: {folder_path}")

    return data_list, labels_list, matrix_sizes


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    # Brain region mapping path
    brain_regions_mapping_file = r'Allen_Brain_Regions(1).xlsx'
    base_input_dir = r'data'
    base_output_dir = r'output/three_receptors_top_contributing_brain_regions_rf'
    os.makedirs(base_output_dir, exist_ok=True)
    try:
        brain_regions, brain_regions_abbr = load_brain_regions_mapping(brain_regions_mapping_file)
        print(f"✅ Brain region mapping loaded: {len(brain_regions)} full names, {len(brain_regions_abbr)} abbreviations")
    except FileNotFoundError as e:
        print(f"Error: brain region mapping file not found: {e}")
        return

    # Define receptor groups
    receptor_groups = {
        'GABA': ['iso', 'propofol'],
        'NMDA': ['ketamine', 'n2o'],
        'α2': ['dex']
    }



    # ---------------- Initialize Logger ----------------
    logger = MetricsLogger()
    metrics_pkl_path = os.path.join(base_output_dir, "metrics_data.pkl")

    # Try loading existing data
    data_loaded = logger.load_data(metrics_pkl_path)
    if data_loaded:
        print(f"✅ Loaded existing performance data: {metrics_pkl_path}")
    # ---------------------------------------------------

    all_receptor_labels = []
    x_axis_labels = None

    # No longer storing data for specificity analysis (moved to a separate script)

    # Iterate each receptor group
    for receptor_name, anesthetic_list in receptor_groups.items():
        print(f"\n{'='*70}")
        print(f"🚀 Start processing receptor: {receptor_name} ({', '.join(anesthetic_list)})")
        print(f"{'='*70}")

        current_output_dir = os.path.join(base_output_dir, f"receptor_{receptor_name}")
        os.makedirs(current_output_dir, exist_ok=True)
        output_prefix = f"receptor_{receptor_name}"

        # ---------------- Resume / checkpoint logic ----------------
        matrix_filename_check = os.path.join(current_output_dir, f"{output_prefix}_rf_feature_importance_matrix.csv")
        results_filename_check = os.path.join(current_output_dir, f"{output_prefix}_rf_model_results.txt")

        # Check whether key output files exist
        if os.path.exists(matrix_filename_check):
            print(f"\n{'='*70}")
            print(f"⏩ Detected existing importance matrix for {receptor_name}; skipping model training...")
            print(f"   File path: {matrix_filename_check}")
            if os.path.exists(results_filename_check):
                print(f"   Results file: {results_filename_check}")
            print(f"{'='*70}\n")

            try:
                # Load importance matrix
                importance_matrix = pd.read_csv(matrix_filename_check, header=None).values
                matrix_size = importance_matrix.shape[0]
                print(f"✅ Successfully loaded importance matrix: {matrix_size}x{matrix_size}")

                # Generate x-axis labels (if not already)
                if x_axis_labels is None:
                    x_axis_labels = [brain_regions.get(i, f'R{i}') for i in range(matrix_size)]

                # Generate abbreviation labels
                x_axis_labels_abbr = [brain_regions_abbr.get(i, f'R{i}') for i in range(matrix_size)]

                # Re-generate visualization (if needed)
                title = f"{receptor_name} Receptor RF Feature Importance Matrix"
                save_prefix = os.path.join(current_output_dir, f"{output_prefix}_rf_feature_importance_matrix")
                plot_importance_heatmap(importance_matrix, x_axis_labels, title, save_prefix, x_axis_labels_abbr)
                print("✅ Heatmap visualization regenerated")

                # Save to summary structure
                all_receptor_labels.append(receptor_name)

                print(f"✅ {receptor_name} receptor loaded (training skipped)\n")
                continue  # Move to next receptor

            except Exception as e:
                print(f"\n{'='*70}")
                print(f"⚠️ Failed to load existing result files: {e}")
                print(f"🔄 Re-training {receptor_name} to generate complete outputs...")
                print(f"{'='*70}\n")
        else:
            print(f"\n📝 Importance matrix not found for {receptor_name}; starting training...")
        # ------------------------------------------------------------

        # === Load anesthetic and non-anesthetic data for the receptor group ===
        print(f"\n📂 Loading anesthetic data for receptor {receptor_name}...")
        anes_data, anes_labels, anes_sizes = load_receptor_data(base_input_dir, anesthetic_list, label=1)

        print(f"📂 Loading non-anesthetic data for receptor {receptor_name}...")
        non_anes_data, non_anes_labels, non_anes_sizes = load_receptor_data(base_input_dir, anesthetic_list, label=0)

        # Merge
        data_list = anes_data + non_anes_data
        labels_list = anes_labels + non_anes_labels
        matrix_sizes = anes_sizes + non_anes_sizes

        if not data_list:
            print(f"⚠️ No data found for receptor {receptor_name}; skipping")
            continue

        matrix_size = matrix_sizes[0]
        print(f"✅ Data loaded for {receptor_name}: anesthetic={len(anes_data)}, non-anesthetic={len(non_anes_data)}, total={len(data_list)}")

        # SMOTE
        if len(set(labels_list)) > 1:
            smote = SMOTE(random_state=GLOBAL_SEED)
            X_resampled, y_resampled = smote.fit_resample(np.array(data_list), labels_list)
        else:
            X_resampled = np.array(data_list)
            y_resampled = labels_list

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=GLOBAL_SEED, stratify=y_resampled
        )

        # Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ============================================================
        # Random Forest hyperparameter search (ref. utils.py)
        # ============================================================
        print(f"\n🔍 Starting Random Forest grid search ({receptor_name})...")

        USE_GRIDSEARCH = True  # Set to False temporarily if you want a quick run
        if USE_GRIDSEARCH:
            rf_base = RandomForestClassifier(random_state=GLOBAL_SEED)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced'],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
            grid_search = GridSearchCV(
                estimator=rf_base,
                param_grid=param_grid,
                cv=cv,
                n_jobs=-1,
                scoring='accuracy',
                verbose=1
            )
            print("🔎 Running GridSearchCV for Random Forest hyperparameters...")
            grid_search.fit(X_train, y_train)
            print(f"✅ GridSearchCV finished. Best params: {grid_search.best_params_}")
            rf = grid_search.best_estimator_
            best_params_str = str(grid_search.best_params_)
        else:
            rf = RandomForestClassifier(random_state=GLOBAL_SEED)
            rf.fit(X_train, y_train)
            best_params_str = "random_state=GLOBAL_SEED (default RF)"

        # Prediction
        y_prob = rf.predict_proba(X_test)[:, 1]
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Random Forest has no evals_result; set history to None
        history = None

        # Log into Logger
        logger.log_result(receptor_name, y_test, y_prob, history)

        print(f"Random Forest test accuracy ({receptor_name}): {accuracy:.4f}")

        # Save text results
        with open(os.path.join(current_output_dir, f"{output_prefix}_rf_model_results.txt"), 'w') as f:
            f.write(f"Receptor type: {receptor_name}\n")
            f.write(f"Included anesthetics: {', '.join(anesthetic_list)}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Params: {best_params_str}\n")

        # --- Build feature-importance matrix and compute region importance ---
        original_feature_importances = rf.feature_importances_
        upper_tri_indices = np.triu_indices(matrix_size, k=1)

        # Build symmetric importance matrix
        importance_matrix = np.zeros((matrix_size, matrix_size))
        importance_matrix[upper_tri_indices] = original_feature_importances
        importance_matrix += importance_matrix.T

        all_receptor_labels.append(receptor_name)

        # Save importance_matrix to CSV
        matrix_csv_path = os.path.join(current_output_dir, f"{output_prefix}_rf_feature_importance_matrix.csv")
        pd.DataFrame(importance_matrix).to_csv(matrix_csv_path, index=False, header=False)

        # Generate brain-region labels and plot heatmaps (full and abbreviated)
        if x_axis_labels is None:
            x_axis_labels = [brain_regions.get(i, f'R{i}') for i in range(matrix_size)]

        # Abbreviation labels
        x_axis_labels_abbr = [brain_regions_abbr.get(i, f'R{i}') for i in range(matrix_size)]

        title = f"{receptor_name} Receptor RF Feature Importance Matrix"
        save_prefix = os.path.join(current_output_dir, f"{output_prefix}_rf_feature_importance_matrix")
        plot_importance_heatmap(importance_matrix, x_axis_labels, title, save_prefix, x_axis_labels_abbr)

        print(f"✅ Finished processing receptor {receptor_name}.")

    # ------------------------------------------------------------------
    # After loop: save data and plot curves
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("📊 Generating standardized performance curves (ROC/PR/Loss)...")
    logger.save_data(metrics_pkl_path)
    plot_all_curves(logger.results, base_output_dir, "All_Receptors_Performance")
    print("="*70 + "\n")

    print("\n🎉 Model training pipeline finished.")
    print("💡 Tip: Run '三个受体分别特异性_RF.py' for inter-receptor specificity analysis")


if __name__ == "__main__":
    main()