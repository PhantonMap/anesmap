import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import warnings
import sys


from metrics_tool import MetricsLogger
from utils_refactored import (
    compute_region_importance_from_matrix,
    load_and_visualize_existing_matrix,
    read_excel_data,
    load_brain_regions_mapping,
    load_raw_data_for_anesthetic
)

# Disable all warnings
warnings.filterwarnings("ignore")

# Global random seed control
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)



def parallel_search(params, search_params):
    """
    Parallel grid search function - Random Forest version
    """
    X_train = search_params["X_train"]
    y_train = np.array(search_params["y_train"], dtype=np.int32)
    cv = search_params["cv"]
    param_keys = search_params["param_keys"]

    current_params = dict(zip(param_keys, params))
    current_params['random_state'] = GLOBAL_SEED
    current_params['n_jobs'] = 1  # Use single thread per process to avoid over-parallelization

    scores = []
    for train_index, val_index in cv.split(X_train, y_train):
        train_index = train_index.astype(int)
        val_index = val_index.astype(int)
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model = RandomForestClassifier(**current_params)
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_pred)
        scores.append(score)

    mean_score = np.mean(scores)
    return mean_score, current_params



def main():
    # Brain region mapping file path
    brain_regions_mapping_file = r'Allen_Brain_Regions(1).xlsx'
    base_input_dir = r'data'
    base_output_dir = r'output/five_anesthetics_contribution_rf'
    try:
        brain_regions, brain_regions_abbr = load_brain_regions_mapping(brain_regions_mapping_file)
        print(f"✅ Brain region mapping loaded: {len(brain_regions)} full names, {len(brain_regions_abbr)} abbreviations")
    except FileNotFoundError as e:
        print(f"Error: Brain region mapping file not found: {e}")
        return


    TRAIN_GROUPS = ['dex', 'iso', 'ketamine', 'n2o', 'propofol', 'all']
    THRESHOLD_MODES = ['top_10', 'top_50']  # Threshold modes for specificity analysis



    os.makedirs(base_output_dir, exist_ok=True)

    logger = MetricsLogger()
    metrics_pkl_path = os.path.join(base_output_dir, "metrics_data.pkl")

    # Try to load existing data
    data_loaded = logger.load_data(metrics_pkl_path)
    if data_loaded:
        print(f"✅ Existing performance data loaded: {metrics_pkl_path}")
    # ---------------------------------------------------------

    all_feature_importances = []
    all_anesthetic_labels = []
    x_axis_labels = None
    region_importances_dict = {}
    raw_data_store = {}
    importance_matrices_dict = {}

    for anes_type in TRAIN_GROUPS:
        print("\n======================================================")
        print(f"🚀 Start processing anesthetic: {anes_type.upper()}")
        print("======================================================")

        current_output_dir = os.path.join(base_output_dir, anes_type)
        output_prefix = f"{anes_type}"

        matrix_filename_check = os.path.join(current_output_dir, f"{output_prefix}_rf_feature_importance_matrix.csv")

        if os.path.exists(matrix_filename_check):
            print(f"⏩ Detected existing complete results for {anes_type}; skipping training and loading directly...")

            try:
                # Load matrix and generate visualization using utils function
                # (use the first threshold_mode as default)
                region_importances, labels = load_and_visualize_existing_matrix(
                    matrix_filename_check, brain_regions, anes_type, current_output_dir, brain_regions_abbr, THRESHOLD_MODES[0]
                )

                all_feature_importances.append(region_importances)
                all_anesthetic_labels.append(anes_type.upper())
                region_importances_dict[anes_type] = region_importances

                # Load importance matrix for specificity analysis (including 'all')
                importance_matrix = pd.read_csv(matrix_filename_check, header=None).values
                importance_matrices_dict[anes_type] = importance_matrix

                # Restore x-axis labels
                if x_axis_labels is None:
                    x_axis_labels = labels

                raw_data, _, pos_count, neg_count = load_raw_data_for_anesthetic(anes_type, base_input_dir)
                if raw_data is not None:
                    raw_data_store[anes_type] = raw_data

                continue  # Move to the next anesthetic
            except Exception as e:
                print(f"⚠️ Failed to load existing result file: {e}")
                print(f"🔄 Will retrain {anes_type} to generate complete results...")

        # === Original data loading and processing pipeline ===
        data_list = []
        labels_list = []
        matrix_sizes = []

        data_folders = {
            1: os.path.join(base_input_dir, f"anesthetic_{anes_type}", "all"),
            0: os.path.join(base_input_dir, f"non_anesthetic_{anes_type}", "all")
        }


        raw_data, _, pos_count, neg_count = load_raw_data_for_anesthetic(anes_type, base_input_dir)
        if raw_data is not None:
            raw_data_store[anes_type] = raw_data
        # ------------------------------------------------------------------

        # File reading loop
        for label, folder_path in data_folders.items():
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

        if not data_list:
            continue

        matrix_size = matrix_sizes[0]

        X_resampled = np.array(data_list)
        y_resampled = np.array(labels_list)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=GLOBAL_SEED, stratify=y_resampled
        )

        # Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        USE_GRIDSEARCH = True

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
            print("🔎 Starting GridSearchCV for Random Forest hyperparameters...")
            grid_search.fit(X_train, y_train)
            print(f"✅ GridSearchCV completed. Best params: {grid_search.best_params_}")
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

        # Random Forest has no training history
        history = None
        logger.log_result(anes_type, y_test, y_prob, history)

        print(f"Random Forest test accuracy ({anes_type}): {accuracy:.4f}")

        # Save text results
        os.makedirs(current_output_dir, exist_ok=True)
        with open(os.path.join(current_output_dir, f"{output_prefix}_rf_model_results.txt"), 'w') as f:
            f.write(f"Anesthetic type: {anes_type.upper()}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Params: {best_params_str}\n")

        # --- Build feature importance matrix and compute region importance ---
        original_feature_importances = rf.feature_importances_
        upper_tri_indices = np.triu_indices(matrix_size, k=1)

        # Build a symmetric importance matrix
        importance_matrix = np.zeros((matrix_size, matrix_size))
        importance_matrix[upper_tri_indices] = original_feature_importances
        importance_matrix += importance_matrix.T

        # Compute brain region importance using utils function
        region_importances = compute_region_importance_from_matrix(importance_matrix)

        # all_feature_importances.append(region_importances)
        all_anesthetic_labels.append(anes_type.upper())
        # region_importances_dict[anes_type] = region_importances


        print(f"⏩ Skipping permutation test. The following threshold modes will be used for specificity analysis: {THRESHOLD_MODES}")
        # ------------------------------------------------------------------

        if x_axis_labels is None:
            x_axis_labels = [brain_regions.get(i, f'R{i}') for i in range(matrix_size)]



        print(f"✅ {anes_type.upper()} processing completed.")



    print("\n🎉 Training and importance matrix generation completed.")
    print("💡 To run specificity analysis, execute: five_anesthetics_contribution_rf.py")

if __name__ == "__main__":
    main()