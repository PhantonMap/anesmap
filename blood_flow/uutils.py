import os
from collections import defaultdict

import pandas as pd
import numpy as np


def get_filenames(root_dir):
    genders = defaultdict(list)
    anesthetic_names = defaultdict(list)
    conditions = {}
    gender_condition = defaultdict(list)
    anesthetic_condition = defaultdict(list)
    gender_anesthetic = defaultdict(list)
    gender_anesthetic_condition = defaultdict(list)

    files = os.listdir(root_dir)
    for file in files:
        if file == 'semi-conscious':
            continue
        excel_files = os.listdir(os.path.join(root_dir, file))
        conditions[file] = [os.path.join(root_dir, file, excel_file) for excel_file in excel_files]
        for excel_file in excel_files:
            file_path = os.path.join(root_dir, file, excel_file)
            head = excel_file.split('_')[0]

            if 'female' in head:
                gender_type = 'female'
            else:
                gender_type = 'male'

            genders[f'{gender_type}'].append(file_path)
            gender_condition[f'{gender_type}_{file}'].append(file_path)

            head = head.split(f'{gender_type}')[0]
            gender_anesthetic[f'{gender_type}_{head}'].append(file_path)

            anesthetic_names[head].append(file_path)
            anesthetic_condition[f'{head}_{file}'].append(file_path)

            gender_anesthetic_condition[f'{gender_type}_{head}_{file}'].append(file_path)

    res = {
        'genders': genders,
        'anesthetics': anesthetic_names,
        'conditions': conditions,
        'gender_anesthetic': gender_anesthetic,
        'gender_condition': gender_condition,
        'anesthetic_condition': anesthetic_condition,
        'gender_anesthetic_condition': gender_anesthetic_condition
    }
    return res


def read_excel(file_path, set_zero=False):
    matrix_list = []
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:

        df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
        df = df.fillna(0)

        matrix = df.to_numpy()

        if set_zero:
            mask = np.tril(np.ones_like(matrix, dtype=bool))
            matrix[mask] = 0

        matrix = matrix[np.newaxis, :, :]
        matrix_list.append(matrix)
    return matrix_list


def filter_and_remap_labels(data, target_labels):
    target_ids = [data['label2id'][label] for label in target_labels]

    mask = np.isin(data['labels'], target_ids)
    filtered_matrix = data['matrix'][mask]
    filtered_labels = data['labels'][mask]

    if len(filtered_labels) == 0:
        raise ValueError

    unique_labels = np.unique(filtered_labels)
    new_label2id = {f'class{label}': i for i, label in enumerate(unique_labels)}
    new_id2label = {i: f'class{label}' for i, label in enumerate(unique_labels)}

    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    remapped_labels = np.array([label_mapping[label] for label in filtered_labels])

    new_data = {
        'matrix': filtered_matrix,
        'labels': remapped_labels,
        'label2id': new_label2id,
        'id2label': new_id2label
    }

    return new_data


def filter_merge_and_remap_labels(data, target_labels, new_categories=None):

    new_labels = np.full_like(data['label'], -1)
    new_label2id = {}
    new_id2label = {}

    for new_class_id, label_group in enumerate(target_labels):
        label_group = [data['label2id'][l] for l in label_group]
        mask = np.isin(data['label'], label_group)
        new_labels[mask] = new_class_id
        if new_categories is not None:
            new_label2id[new_categories[new_class_id]] = new_class_id
            new_id2label[new_class_id] = new_categories[new_class_id]
        else:
            new_label2id[f'group{new_class_id}'] = new_class_id
            new_id2label[new_class_id] = f'group{new_class_id}'

    valid_mask = new_labels != -1
    filtered_matrix = data['data'][valid_mask]
    filtered_labels = new_labels[valid_mask]

    new_data = {
        'matrix': filtered_matrix,
        'labels': filtered_labels,
        'label2id': new_label2id,
        'id2label': new_id2label
    }

    return new_data