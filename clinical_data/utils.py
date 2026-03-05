import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_processed_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file {file_path} do not exist, run process_and_save_data() first")

    data_dict = np.load(file_path, allow_pickle=True).item()
    return data_dict


def get_data_by_labels(include_labels, data_dict):
    all_data = data_dict['data']
    all_labels = data_dict['label']
    label2id = data_dict['label2id']

    valid_labels = set(label2id.keys())
    for label in include_labels:
        if label not in valid_labels:
            raise ValueError(f"Invalid label: {label}. Valid labels are: {valid_labels}")

    include_ids = [label2id[label] for label in include_labels]

    mask = np.isin(all_labels, include_ids)

    selected_data = all_data[mask]
    selected_labels = all_labels[mask]

    return selected_data


def random_forest_classification_with_gridsearch(data, labels,
                                                 feature_names=None,
                                                 test_size=0.2,
                                                 random_state=42,
                                                 normalization='none',
                                                 save_name=None):
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    if normalization == 'standard':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif normalization == 'minmax':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        pass

    rf = RandomForestClassifier(random_state=random_state)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced'],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("\nacc:", accuracy_score(y_test, y_pred))
    if save_name is not None:
        with open(save_name+'_acc.txt', 'w') as f:
            f.write(str(accuracy_score(y_test, y_pred)))

    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()

    feature_importance = best_model.feature_importances_

    plot_feature_importance_heatmap(feature_importance, feature_names, save_name)

    return best_model, feature_importance


def plot_feature_importance_heatmap(feature_importance, feature_names, save_name=None):
    importance_softmax = softmax(feature_importance)

    importance_df = pd.DataFrame({
        'BrainRegion': feature_names,
        'Importance': importance_softmax
    })

    heatmap_data = importance_softmax.reshape(1, -1)

    plt.figure(figsize=(50, 20))
    sns.heatmap(
        heatmap_data,
        annot=False,
        fmt=".3f",
        cmap=plt.cm.coolwarm,
        xticklabels=feature_names,
        yticklabels=False,
        cbar=True
    )

    plt.title("Brain Region Importance (Softmax Normalized)", pad=20)
    plt.xticks(rotation=90, ha='center')
    plt.xlabel("Brain Regions")
    # plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name+'.png')
        plt.savefig(save_name+'.pdf')
    plt.show()


def split_brain_regions(base_names, input_data):
    left_indices = []
    right_indices = []
    bilateral_indices = []

    left_names = []
    right_names = []

    for i, name in enumerate(base_names):
        if name.endswith('_L'):
            left_indices.append(i)
            left_names.append(name[:-2])
        elif name.endswith('_R'):
            right_indices.append(i)
            right_names.append(name[:-2])
        else:
            bilateral_indices.append(i)
            left_names.append(name)
            right_names.append(name)

    combined_left_indices = sorted(left_indices + bilateral_indices)
    combined_right_indices = sorted(right_indices + bilateral_indices)

    left_data = input_data[:, combined_left_indices]
    right_data = input_data[:, combined_right_indices]

    for i, j in zip(left_names, right_names):
        if i != j:
            print(f'The {i} and {j} brain regions do not match.')

    return left_data, right_data, left_names

