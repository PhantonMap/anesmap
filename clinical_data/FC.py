import os.path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_processed_data, get_data_by_labels


def filter_brain_regions(data, brain_regions):
    left_indices = [i for i, region in enumerate(brain_regions) if region.endswith('_L')]
    right_indices = [i for i, region in enumerate(brain_regions) if region.endswith('_R')]
    both_indices = [i for i, region in enumerate(brain_regions) if not (region.endswith('_L') or region.endswith('_R'))]

    stripped_regions = []
    seen = set()
    for region in brain_regions:
        stripped = region[:-2] if region.endswith(('_L', '_R')) else region
        if stripped not in seen:
            seen.add(stripped)
            stripped_regions.append(stripped)

    left_regions = left_indices + both_indices
    left_regions = sorted(left_regions)
    left_data = data[:, left_regions, :][:, :, left_regions]

    right_regions = right_indices + both_indices
    right_regions.sort()
    right_data = data[:, right_regions, :][:, :, right_regions]

    return left_data, right_data, stripped_regions


def extract_upper_triangular(data):
    n_subjects, n_nodes, _ = data.shape
    upper_indices = np.triu_indices(n_nodes, k=1)
    features = []
    for subject in data:
        upper_values = subject[upper_indices]
        features.append(upper_values)
    return np.array(features)


def reconstruct_from_upper_triangular(upper_values, dim):
    batch_size = upper_values.shape[0]

    matrices = np.zeros((batch_size, dim, dim))

    upper_indices = np.triu_indices(dim, k=1)

    matrices[:, upper_indices[0], upper_indices[1]] = upper_values

    matrices[:, upper_indices[1], upper_indices[0]] = upper_values

    return matrices


def generate_mask(matrix, topN):
    topN_indices = np.argsort(matrix, axis=None)[-topN:]
    mask = np.zeros_like(matrix, dtype=bool)
    mask.flat[topN_indices] = True
    return mask


def write_non_zero(meanvalues_df, var_df, save_path):
    non_zero_data = []

    for i, (xlabel, row) in enumerate(meanvalues_df.iterrows()):
        for j, (ylabel, value) in enumerate(row.items()):
            if value > 0:
                # print(xlabel, ylabel, value)
                non_zero_data.append({'xlabel': xlabel, 'ylabel': ylabel, 'value': value, 'var': var_df.iloc[i, j]})

    non_zero_df = pd.DataFrame(non_zero_data)

    non_zero_df.to_excel(save_path, index=False)


def save_numpy_to_excel(data, row_names, col_names, file_path):
    if len(row_names) != data.shape[0] or len(col_names) != data.shape[1]:
        raise ValueError("The length of the column name must match the data dimension!")

    df = pd.DataFrame(data, index=row_names, columns=col_names)

    df.to_excel(file_path, index=True)


def save_original_data(data, save_path):
    data = reconstruct_from_upper_triangular(data, 62)
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        for i in range(data.shape[0]):
            df = pd.DataFrame(data[i, :, :], index=brain_regions, columns=brain_regions)
            df.to_excel(writer, sheet_name=f"Subject_{i}")


if __name__ == '__main__':
    save_dir = "./results/FC"
    os.makedirs(save_dir, exist_ok=True)
    base_path = "data/fc"
    mcs_data = load_processed_data(os.path.join(base_path, "MCS.npy"))
    sub_data = load_processed_data(os.path.join(base_path, "Sub.npy"))
    uws_data = load_processed_data(os.path.join(base_path, "UWS.npy"))

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_excel('./data/fc/brain_region.xlsx', sheet_name='Sheet1', header=0)
    brain_regions_all = df.iloc[:, 0].tolist()

    sub_left, sub_right, brain_regions = filter_brain_regions(sub_data["data"], brain_regions_all)
    sub_left = extract_upper_triangular(sub_left)
    sub_right = extract_upper_triangular(sub_right)
    sub_data["data"] = np.concatenate((sub_left, sub_right), axis=0)
    sub_data["label"] = np.concatenate((sub_data["label"], sub_data["label"]), axis=0)
    sub_data = get_data_by_labels(['Sub'], sub_data)

    uws_left, uws_right, _ = filter_brain_regions(uws_data["data"], brain_regions_all)
    uws_left = extract_upper_triangular(uws_left)
    uws_data["data"] = uws_left
    uws_left = get_data_by_labels(['UWS_L', 'UWS_LR'], uws_data)
    uws_right = extract_upper_triangular(uws_right)
    uws_data["data"] = uws_right
    uws_right = get_data_by_labels(['UWS_R', 'UWS_LR'], uws_data)
    uws_data = np.concatenate((uws_left, uws_right), axis=0)

    mcs_left, mcs_right, _ = filter_brain_regions(mcs_data["data"], brain_regions_all)
    mcs_left = extract_upper_triangular(mcs_left)
    mcs_data["data"] = mcs_left
    mcs_left = get_data_by_labels(['MCS_L', 'MCS_LR'], mcs_data)
    mcs_right = extract_upper_triangular(mcs_right)
    mcs_data["data"] = mcs_right
    mcs_right = get_data_by_labels(['MCS_R', 'MCS_LR'], mcs_data)
    mcs_data = np.concatenate((mcs_left, mcs_right), axis=0)

    datas = {
        "mcs": mcs_data,  # [n, dim]
        "sub": sub_data,  # [n, dim]
        "uws": uws_data,  # [n, dim]
        "bothMcsUws": np.concatenate((mcs_data, uws_data), axis=0),
    }

    top_masks = {}
    for a, b in [("mcs", "sub"), ("uws", "sub"), ("mcs", "uws"), ("sub", "bothMcsUws")]:
        print("*"*50)
        a_features = datas[a]
        b_features = datas[b]

        a_labels = np.zeros(len(a_features))
        b_labels = np.ones(len(b_features))

        X = np.vstack([a_features, b_features])
        y = np.concatenate([a_labels, b_labels])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced']
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)

        print("best params:", grid_search.best_params_)
        best_rf = grid_search.best_estimator_

        y_pred = best_rf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=[a, b]))  # , 'UWS'
        print(f"acc: {accuracy_score(y_test, y_pred):.4f}")
        with open(f'{save_dir}/{a}AND{b}_accuracy.txt', 'w') as f:
            f.write(str(accuracy_score(y_test, y_pred)))

        plt.figure(figsize=(10, 7))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        n_nodes = len(brain_regions)
        feature_importance = best_rf.feature_importances_

        importance_matrix = np.zeros((n_nodes, n_nodes))
        upper_indices = np.triu_indices(n_nodes, k=1)
        importance_matrix[upper_indices] = feature_importance

        importance_matrix = importance_matrix + importance_matrix.T

        red_blue_cmap = sns.diverging_palette(240, 10, as_cmap=True)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            importance_matrix,
            cmap=plt.cm.coolwarm,
            square=True,
            xticklabels=brain_regions,
            yticklabels=brain_regions,
            cbar_kws={"label": "Feature Importance"}
        )
        plt.title("Feature Importance Matrix (Random Forest)")
        plt.tight_layout()

        plt.savefig(f'{save_dir}/{a}AND{b}_importanceMatrix.png')
        plt.savefig(f'{save_dir}/{a}AND{b}_importanceMatrix.pdf')
        plt.show()

        save_numpy_to_excel(importance_matrix, brain_regions, brain_regions, f'{save_dir}/{a}AND{b}_importanceMatrix.xlsx')

        mask = generate_mask(importance_matrix, 10)
        top_masks[f"{a}_{b}"] = generate_mask(importance_matrix, 100)

        for condition in ["mcs", "sub", "uws"]:
            mean_values = np.ma.masked_array(reconstruct_from_upper_triangular(datas[condition], n_nodes).mean(axis=0), ~mask)
            var_values = np.ma.masked_array(reconstruct_from_upper_triangular(datas[condition], n_nodes).var(axis=0), ~mask)

            mean_values_df = pd.DataFrame(mean_values, index=brain_regions, columns=brain_regions)
            var_values_df = pd.DataFrame(var_values, index=brain_regions, columns=brain_regions)
            write_non_zero(mean_values_df, var_values_df, f'{save_dir}/{a}AND{b}_mean_values_{condition}.xlsx')

    mcs_sub_mask = top_masks["mcs_sub"]
    uws_sub_mask = top_masks["uws_sub"]

    intersection = mcs_sub_mask & uws_sub_mask
    mcs_only = mcs_sub_mask & ~uws_sub_mask
    uws_only = uws_sub_mask & ~mcs_sub_mask

    for condition in ["mcs", "sub", "uws"]:
        for mask_name, mask in [("intersection", intersection), ("only_mcsANDsub", mcs_only), ("only_uwsANDsub", uws_only)]:
            mean_values = np.ma.masked_array(reconstruct_from_upper_triangular(datas[condition], n_nodes).mean(axis=0), ~mask)
            var_values = np.ma.masked_array(reconstruct_from_upper_triangular(datas[condition], n_nodes).var(axis=0), ~mask)

            mean_values_df = pd.DataFrame(mean_values, index=brain_regions, columns=brain_regions)
            var_values_df = pd.DataFrame(var_values, index=brain_regions, columns=brain_regions)
            write_non_zero(mean_values_df, var_values_df, f'{save_dir}/{mask_name}_{condition}.xlsx')