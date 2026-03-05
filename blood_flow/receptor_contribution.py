import os.path

import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier

from uutils import filter_merge_and_remap_labels

np.random.seed(42)


def random_forest_classification(data):
    X = data['matrix']
    y = data['labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    feature_importances = rf.feature_importances_

    accuracy = rf.score(X_test, y_test)

    n_permutations = 1
    permuted_importances = np.zeros((n_permutations, len(feature_importances)))

    with tqdm(range(n_permutations), desc="Permutation Test", unit="iteration") as pbar:
        for i in pbar:
            permuted_y_train = np.random.permutation(y_train)
            permuted_rf = RandomForestClassifier(random_state=42)
            permuted_rf.fit(X_train, permuted_y_train)
            permuted_importances[i] = permuted_rf.feature_importances_

    p_values = []
    for j in range(len(feature_importances)):
        comparison = permuted_importances[:, j] >= feature_importances[j]
        p = np.mean(comparison)
        p_values.append(p)
    p_values = np.array(p_values)

    alpha = 0.05
    significant_features = p_values < alpha

    feature_importances = np.where(significant_features, feature_importances, 0)

    return feature_importances, accuracy


def xgboost_classification(data):
    X = data['matrix']
    y = data['labels']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        reg_alpha=0,
        reg_lambda=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        tree_method='hist',
        device='cuda:0',
        random_state=42,
        scale_pos_weight=1
    )

    xgb_model.fit(X_train, y_train)

    feature_importances = xgb_model.feature_importances_

    accuracy = xgb_model.score(X_test, y_test)

    n_permutations = 0
    permuted_importances = np.zeros((n_permutations, len(feature_importances)))
    with tqdm(range(n_permutations), desc="Permutation Test", unit="iteration") as pbar:
        for i in pbar:
            permuted_y_train = np.random.permutation(y_train)
            permuted_xgb = XGBClassifier(
                max_depth=3,
                learning_rate=0.1,
                n_estimators=100,
                reg_alpha=0,
                reg_lambda=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1,
                tree_method='hist',
                device='cuda:0',
                random_state=42,
                scale_pos_weight=1
            )
            permuted_xgb.fit(X_train, permuted_y_train)
            permuted_importances[i] = permuted_xgb.feature_importances_

    p_values = []
    for j in range(len(feature_importances)):
        comparison = permuted_importances[:, j] >= feature_importances[j]
        p = np.mean(comparison)
        p_values.append(p)
    p_values = np.array(p_values)

    alpha = 0.05
    significant_features = p_values < alpha

    # feature_importances = np.where(significant_features, feature_importances, 0)
    # feature_importances = np.where(feature_importances > 0, 1, 0)
    print('accuracy:', accuracy)
    return feature_importances, accuracy


def plot_feature_importance(feature_importances, index_names, label_names, save_name, N, mode='topN_and_greater_than_zero'):
    row_min = feature_importances.min(axis=1, keepdims=True)
    row_max = feature_importances.max(axis=1, keepdims=True)
    feature_importances = (feature_importances - row_min) / (row_max - row_min + 1e-9)

    num = feature_importances.shape[0]
    plt.figure(figsize=(50, 10))
    df = pd.DataFrame(feature_importances, index=index_names, columns=label_names)
    df.to_excel(save_name + '.xlsx')

    sns.heatmap(df, annot=False, cmap=plt.cm.coolwarm)

    if N > 0:
        if mode == 'greater_than_zero':
            for i in range(num):
                for j in range(feature_importances.shape[1]):
                    if feature_importances[i, j] > 0:
                        plt.scatter(j + 0.5, i + 0.5, color='black', marker='*', s=20)
        elif mode == 'topN_and_greater_than_zero' and N > 0:
            top_10_indices = np.argsort(feature_importances, axis=1)[:, -N:]
            for i in range(num):
                for j in top_10_indices[i]:
                    if feature_importances[i, j] > 0:
                        plt.scatter(j + 0.5, i + 0.5, color='black', marker='*', s=20)

    plt.xlabel('Brain Regions')
    plt.ylabel('Conditions')
    plt.title('Brain Blood Flow Contribution Heatmap')

    plt.subplots_adjust(bottom=0.5)

    plt.savefig(save_name + '.pdf', format="pdf")
    plt.savefig(save_name + '.png')
    plt.close()


def save_top_features(data, save_name):
    top_10_indices = np.argsort(data)[-10:][::-1]
    top_10_values = data[top_10_indices]

    top_10_dict = {
        "indices": top_10_indices.tolist(),
        "values": top_10_values.tolist()
    }

    with open(save_name, 'w') as f:
        json.dump(top_10_dict, f)


def save_topN(data_list, xlabel, group_labels, save_name, N):
    topN_labels = {}
    for data, group_label in zip(data_list, group_labels):
        top_10_indices = np.argsort(data)[-N:][::-1]
        top_10_labels = [xlabel[i] for i in top_10_indices]
        topN_labels[group_label] = top_10_labels
    # df = pd.DataFrame(topN_labels)
    df = pd.DataFrame.from_dict(topN_labels, orient="index")
    columns = ["Name"] + [f"Top{i + 1}" for i in range(N)]

    df.reset_index(inplace=True)
    df.columns = columns

    df.to_excel(save_name, header=False)


if __name__ == '__main__':
    label_def = pd.read_excel(r'data/brain_regions.xlsx')
    label_names = label_def['Brain_Region'].tolist()

    topN = 50

    # method = 'xgboost'
    method = 'random_forest'
    methods = {
        'xgboost': xgboost_classification,
        'random_forest': random_forest_classification
    }

    data_path = 'data/brain1.npy'
    save_dir = './results/receptor_contribution'
    os.makedirs(save_dir, exist_ok=True)
    all_data = np.load(data_path, allow_pickle=True).item()

    groups = [
        {
            'group': 'all',
            'tasks': [
                {
                    'classification_type': '2',
                    'task': 'α2-Anesthetic/Conscious',
                    'target_labels': [['dex_conscious_female', 'dex_conscious_male'],
                                      ['dex_anesthesia_female', 'dex_anesthesia_male']]
                },
                {
                    'classification_type': '2',
                    'task': 'GABA-Anesthetic/Conscious',
                    'target_labels': [['iso_conscious_female', 'propofol_conscious_female', 'iso_conscious_male', 'propofol_conscious_male'],
                                      ['iso_anesthesia_female', 'propofol_anesthesia_female', 'iso_anesthesia_male', 'propofol_anesthesia_male']]
                },
                {
                    'classification_type': '2',
                    'task': 'NMDA-Anesthetic/Conscious',
                    'target_labels': [['ketamine_conscious_female', 'N2O_conscious_female', 'ketamine_conscious_male', 'N2O_conscious_male'],
                                      ['ketamine_anesthesia_female', 'N2O_anesthesia_female', 'ketamine_anesthesia_male', 'N2O_anesthesia_male']]
                },
            ]
        }

    ]

    for group in groups:
        group_name = group['group']
        if not os.path.exists(os.path.join(save_dir, group_name)):
            os.mkdir(os.path.join(save_dir, group_name))
        configs = group['tasks']

        group_feature_importances = []
        group_labels = []

        for config in tqdm(configs):
            task_name = config['task'].replace('/', '_')
            print(task_name)
            if os.path.exists(os.path.join(save_dir, group_name, task_name, 'imp_feature.npy')):
                group_feature_importances.append(
                    np.load(os.path.join(save_dir, group_name, task_name, 'imp_feature.npy')))
                group_labels.append(task_name.split('-')[0])
                continue
            data = filter_merge_and_remap_labels(all_data, target_labels=config['target_labels'])

            feature_importances, accuracy = methods[method](data)

            if not os.path.exists(os.path.join(save_dir, group_name, task_name)):
                os.mkdir(os.path.join(save_dir, group_name, task_name))

            with open(os.path.join(save_dir, group_name, task_name, 'accuracy.txt'), 'w') as f:
                f.write(f'{accuracy}')

            output_json_file = os.path.join(save_dir, group_name, task_name, f'top_{topN}.json')
            save_top_features(feature_importances, output_json_file)

            np.save(os.path.join(save_dir, group_name, task_name, 'imp_feature.npy'), feature_importances)

            group_feature_importances.append(feature_importances)
            group_labels.append(task_name.split('-')[0])

        save_name = os.path.join(save_dir, group_name, 'group_imp')
        if topN > 0:
            save_topN(group_feature_importances, label_names, group_labels,
                      os.path.join(save_dir, group_name, f'top{topN}.xlsx'), N=topN)
        plot_feature_importance(np.stack(group_feature_importances, axis=0), index_names=group_labels,
                                label_names=label_names, save_name=save_name, N=topN)