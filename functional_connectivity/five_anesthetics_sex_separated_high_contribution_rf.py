import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import warnings

from metrics_tool import MetricsLogger, plot_all_curves
from utils_refactored import (
    plot_importance_heatmap,
    load_and_visualize_existing_matrix,
    analyze_by_receptor,
    compute_region_importance_from_matrix,
    read_excel_data,
    load_brain_regions_mapping
)

# Disable all warnings
warnings.filterwarnings("ignore")

# Brain region mapping file path
brain_regions_mapping_file = r'Allen_Brain_Regions(1).xlsx'
base_input_dir = r'data'
base_output_dir = r'output/five_anesthetics_sex_separated_high_contribution_rf'
# ============================================================
# Global random seed
# ============================================================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


def _resolve_path(p: str, base_dir: str) -> str:

    if not p:
        return base_dir
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))


def main():
    # Use the script directory as the base for relative paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Resolve the three input paths
    brain_regions_mapping_path = _resolve_path(brain_regions_mapping_file, SCRIPT_DIR)
    base_input_dir_abs = _resolve_path(base_input_dir, SCRIPT_DIR)
    base_output_dir_abs = _resolve_path(base_output_dir, SCRIPT_DIR)

    # Load brain region mapping
    try:
        brain_regions, brain_regions_abbr = load_brain_regions_mapping(brain_regions_mapping_path)
        print(f"✅ Brain-region mapping loaded: {len(brain_regions)} full names, {len(brain_regions_abbr)} abbreviations")
        print(f"   Mapping file: {brain_regions_mapping_path}")
    except FileNotFoundError as e:
        print(f"Error: brain-region mapping file not found: {e}")
        return

    ANESTHETICS_5 = ['dex', 'iso', 'ketamine', 'n2o', 'propofol']
    TRAIN_GROUPS = ANESTHETICS_5 + ['all']
    SPECIFICITY_GROUPS = TRAIN_GROUPS

    genders = ['male', 'female']

    for gender in genders:
        print("\n" + "=" * 80)
        print(f"🚀🚀🚀 Start processing gender group: {gender.upper()} 🚀🚀🚀")
        print("=" * 80 + "\n")

        # Output directory: rooted at base_output_dir
        gender_output_root = os.path.join(base_output_dir_abs, 'task1_gender_RF', gender)
        os.makedirs(gender_output_root, exist_ok=True)

        # Logger
        logger = MetricsLogger()
        metrics_pkl_path = os.path.join(gender_output_root, "metrics_data.pkl")

        data_loaded = logger.load_data(metrics_pkl_path)
        if data_loaded:
            print(f"✅ Loaded existing performance data: {metrics_pkl_path}")

        all_feature_importances = []
        all_anesthetic_labels = []
        x_axis_labels = None
        region_importances_dict = {}

        raw_data_store = {}            # {anes_type: np.array(n_samples, n_features)} (before SMOTE)
        importance_matrices_dict = {}  # {anes_type: importance_matrix}

        THRESHOLD_MODES = ['top_10', 'top_50']

        for anes_type in TRAIN_GROUPS:
            print("\n======================================================")
            print(f"🚀 Start processing anesthetic: {anes_type.upper()} ({gender.upper()})")
            print("======================================================")

            current_output_dir = os.path.join(gender_output_root, anes_type)
            output_prefix = f"{anes_type}"
            os.makedirs(current_output_dir, exist_ok=True)

            matrix_filename_check = os.path.join(current_output_dir, f"{output_prefix}_rf_feature_importance_matrix.csv")

            # ---------------------------------------------------------
            # If a complete result file already exists:
            # - load and visualize it directly
            # - also preload raw_data (for later specificity analysis)
            # ---------------------------------------------------------
            if os.path.exists(matrix_filename_check):
                print(f"⏩ Found complete result file for {anes_type}; skip training and load directly...")

                try:
                    region_importances, labels = load_and_visualize_existing_matrix(
                        matrix_filename_check, brain_regions, anes_type, current_output_dir, brain_regions_abbr
                    )

                    all_feature_importances.append(region_importances)
                    all_anesthetic_labels.append(anes_type.upper())
                    region_importances_dict[anes_type] = region_importances

                    importance_matrix = pd.read_csv(matrix_filename_check, header=None).values
                    if anes_type in ['all']:
                        importance_matrices_dict[anes_type] = importance_matrix

                    if x_axis_labels is None:
                        x_axis_labels = labels

                    # Preload raw data (pos + neg)
                    temp_list = []

                    pos_folder = os.path.join(base_input_dir_abs, f"anesthetic_{anes_type}", gender)
                    if os.path.exists(pos_folder):
                        files = sorted([file for file in os.listdir(pos_folder) if file.endswith('.xlsx')])
                        for file in files:
                            file_path = os.path.join(pos_folder, file)
                            try:
                                features, _, _ = read_excel_data(file_path, 1)
                                temp_list.append(features)
                            except Exception:
                                continue
                    pos_count = len(temp_list)

                    neg_folder = os.path.join(base_input_dir_abs, f"non_anesthetic_{anes_type}", gender)
                    if os.path.exists(neg_folder):
                        files = sorted([file for file in os.listdir(neg_folder) if file.endswith('.xlsx')])
                        for file in files:
                            file_path = os.path.join(neg_folder, file)
                            try:
                                features, _, _ = read_excel_data(file_path, 0)
                                temp_list.append(features)
                            except Exception:
                                continue
                    neg_count = len(temp_list) - pos_count

                    if temp_list and (anes_type in ANESTHETICS_5):
                        raw_data_store[anes_type] = np.array(temp_list)
                        print(f"✅ Loaded raw data for {anes_type} ({gender}): {len(temp_list)} samples (pos={pos_count}, neg={neg_count})")

                    continue
                except Exception as e:
                    print(f"⚠️ Failed to load existing result file: {e}")
                    print(f"🔄 Will retrain {anes_type} to regenerate complete results...")

            # ---------------------------------------------------------
            # === Training pipeline ===
            # ---------------------------------------------------------
            data_list = []
            labels_list = []
            matrix_sizes = []

            data_folders = {
                1: os.path.join(base_input_dir_abs, f"anesthetic_{anes_type}", gender),
                0: os.path.join(base_input_dir_abs, f"non_anesthetic_{anes_type}", gender)
            }

            # Preload raw data (for specificity analysis)
            all_data_list = []

            pos_folder = data_folders[1]
            if os.path.exists(pos_folder):
                files = sorted([f for f in os.listdir(pos_folder) if f.endswith('.xlsx')])
                for file in files:
                    file_path = os.path.join(pos_folder, file)
                    try:
                        features, _, _ = read_excel_data(file_path, 1)
                        all_data_list.append(features)
                    except Exception:
                        continue
            pos_count = len(all_data_list)

            neg_folder = data_folders[0]
            if os.path.exists(neg_folder):
                files = sorted([f for f in os.listdir(neg_folder) if f.endswith('.xlsx')])
                for file in files:
                    file_path = os.path.join(neg_folder, file)
                    try:
                        features, _, _ = read_excel_data(file_path, 0)
                        all_data_list.append(features)
                    except Exception:
                        continue
            neg_count = len(all_data_list) - pos_count

            if all_data_list and (anes_type in SPECIFICITY_GROUPS):
                raw_data_store[anes_type] = np.array(all_data_list)
                print(f"✅ Preloaded raw data for {anes_type} ({gender}): {len(all_data_list)} samples (pos={pos_count}, neg={neg_count})")

            # Load training data
            for label, folder_path in data_folders.items():
                if os.path.exists(folder_path):
                    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.xlsx')])
                    desc = f"Loading {anes_type} (label {label}, {gender})"
                    for file in tqdm(files, desc=desc, unit="file"):
                        file_path = os.path.join(folder_path, file)
                        try:
                            features, _, n = read_excel_data(file_path, label)
                            data_list.append(features)
                            labels_list.append(label)
                            matrix_sizes.append(n)
                        except Exception:
                            continue

            if not data_list:
                print(f"⚠️ No data found for {anes_type} ({gender}); skipping...")
                continue

            unique_classes = sorted(set(labels_list))
            if len(unique_classes) < 2:
                print(f"⚠️ {anes_type} ({gender}) contains only one class {unique_classes}; cannot train a binary classifier. Skipping.")
                continue

            matrix_size = matrix_sizes[0]

            # SMOTE
            if len(set(labels_list)) > 1:
                try:
                    smote = SMOTE(random_state=GLOBAL_SEED)
                    X_resampled, y_resampled = smote.fit_resample(np.array(data_list), labels_list)
                except Exception as e:
                    print(f"⚠️ SMOTE failed ({anes_type}, {gender}): {e}")
                    print("   Skipping SMOTE and continuing with original samples.")
                    X_resampled = np.array(data_list)
                    y_resampled = np.array(labels_list)
            else:
                X_resampled = np.array(data_list)
                y_resampled = labels_list

            y_resampled = np.array(y_resampled)

            classes_after, counts_after = np.unique(y_resampled, return_counts=True)
            if len(classes_after) < 2:
                print(f"⚠️ {anes_type} ({gender}) still has only one class after resampling; skipping.")
                continue
            if counts_after.min() < 2:
                print(f"⚠️ {anes_type} ({gender}) has too few samples per class {dict(zip(classes_after, counts_after))}; cannot reliably split train/test. Skipping.")
                continue

            # Train/test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_resampled, y_resampled, test_size=0.2, random_state=GLOBAL_SEED, stratify=y_resampled
                )
            except ValueError as e:
                print(f"⚠️ train_test_split(with stratify) failed ({anes_type}, {gender}): {e}")
                print("   Trying split without stratify.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_resampled, y_resampled, test_size=0.2, random_state=GLOBAL_SEED, stratify=None
                )

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print(f"⚠️ {anes_type} ({gender}) has insufficient class diversity after splitting; skipping.")
                continue

            # Standardization
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Random Forest hyperparameter search
            print(f"🔍 Starting Random Forest hyperparameter search ({anes_type}, {gender})...")

            USE_GRIDSEARCH = True
            train_classes, train_counts = np.unique(y_train, return_counts=True)
            min_train_count = int(train_counts.min()) if len(train_counts) else 0

            rf = None
            best_params_str = ""

            if USE_GRIDSEARCH:
                if min_train_count < 2:
                    print(f"⚠️ Too few training samples (min_class_count={min_train_count}); skipping GridSearchCV and using default RF.")
                    rf = RandomForestClassifier(random_state=GLOBAL_SEED)
                    rf.fit(X_train, y_train)
                    best_params_str = "random_state=GLOBAL_SEED (default RF; gridsearch skipped due to small sample)"
                else:
                    rf_base = RandomForestClassifier(random_state=GLOBAL_SEED)
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, None],
                        'min_samples_split': [2, 5, 10],
                        'class_weight': ['balanced'],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2']
                    }
                    n_splits = min(5, min_train_count)
                    if n_splits < 2:
                        print(f"⚠️ Too few training samples for GridSearchCV (min_class_count={min_train_count}); using default RF.")
                        rf = RandomForestClassifier(random_state=GLOBAL_SEED)
                        rf.fit(X_train, y_train)
                        best_params_str = "random_state=GLOBAL_SEED (default RF; gridsearch skipped due to small sample)"
                    else:
                        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED)
                        grid_search = GridSearchCV(
                            estimator=rf_base,
                            param_grid=param_grid,
                            cv=cv,
                            n_jobs=-1,
                            scoring='accuracy',
                            verbose=1
                        )
                        print(f"🔎 Running GridSearchCV for Random Forest hyperparameters... (n_splits={n_splits})")
                        grid_search.fit(X_train, y_train)
                        print(f"✅ GridSearchCV done. Best parameters: {grid_search.best_params_}")
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

            history = None
            logger.log_result(anes_type, y_test, y_prob, history)

            print(f"Random Forest test accuracy ({anes_type}, {gender}): {accuracy:.4f}")

            # Build feature-importance matrix
            original_feature_importances = rf.feature_importances_
            upper_tri_indices = np.triu_indices(matrix_size, k=1)

            importance_matrix = np.zeros((matrix_size, matrix_size))
            importance_matrix[upper_tri_indices] = original_feature_importances
            importance_matrix += importance_matrix.T

            all_anesthetic_labels.append(anes_type.upper())

            print(f"⏩ Skipping permutation test; will perform specificity analysis using the following threshold modes: {THRESHOLD_MODES}")

            matrix_csv_path = os.path.join(current_output_dir, f"{output_prefix}_rf_feature_importance_matrix.csv")
            pd.DataFrame(importance_matrix).to_csv(matrix_csv_path, index=False, header=False)

            # Plot heatmaps
            if x_axis_labels is None:
                x_axis_labels = [brain_regions.get(i, f'R{i}') for i in range(matrix_size)]
            x_axis_labels_abbr = [brain_regions_abbr.get(i, f'R{i}') for i in range(matrix_size)]

            title = f"{anes_type.upper()} RF Feature Importance Matrix ({gender.upper()})"
            save_prefix = os.path.join(current_output_dir, f"{output_prefix}_rf_feature_importance_matrix")
            plot_importance_heatmap(importance_matrix, x_axis_labels, title, save_prefix, x_axis_labels_abbr)

            print(f"✅ {anes_type.upper()} ({gender.upper()}) completed.")

        # Performance curves
        print("\n======================================================")
        print(f"📊 Generating standardized performance curves (ROC/PR/Loss) - {gender.upper()}...")
        logger.save_data(metrics_pkl_path)
        plot_all_curves(logger.results, gender_output_root, f"{gender.upper()}_All_Anesthetics_Performance_RF")
        print("======================================================\n")

        # Receptor-group analysis
        if region_importances_dict and x_axis_labels:
            print("\n======================================================")
            print(f"🧠 Ranking brain-region contributions by receptor group + Kruskal-Wallis + Dunn test ({gender.upper()})...")
            analyze_by_receptor(region_importances_dict, x_axis_labels, gender_output_root, top_k=20, alpha=0.05)
            print("======================================================\n")
        else:
            print(f"⚠️ Cannot run receptor-group analysis ({gender.upper()}): region_importances_dict or x_axis_labels is empty.")


if __name__ == "__main__":
    main()