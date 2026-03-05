import os

import numpy as np
import pandas as pd

from utils import load_processed_data, get_data_by_labels, random_forest_classification_with_gridsearch, \
    split_brain_regions


def save_original_data(data, save_path):
    subject_ids = [f'subject_{i}' for i in range(1, data.shape[0] + 1)]
    df = pd.DataFrame(data, columns=brain_regions, index=subject_ids)
    df.to_excel(save_path, index_label='Subject_ID')


if __name__ == '__main__':
    df = pd.read_excel(r'./data/resting_state/brain_region.xlsx', sheet_name='Sheet1', header=0)
    brain_regions_all = df.iloc[:, 0].tolist()

    save_dir = r'./results/resting_state'
    os.makedirs(save_dir, exist_ok=True)
    for data_type in ['fALFF', 'ALFF', 'ReHo']:
        data_batch1 = load_processed_data(f'./results/resting_state/resting_state-{data_type}-batch1.npy')
        data_batch2 = load_processed_data(f'./results/resting_state/resting_state-{data_type}-batch2.npy')
        data_batch3_mcs = load_processed_data(f'./results/resting_state/MCS-{data_type}-batch3.npy')
        data_batch3_uws = load_processed_data(f'./results/resting_state/UWS-{data_type}-batch3.npy')

        data_task = data_batch1
        data_task['data'] = np.concatenate([data_task['data'], data_batch2['data'], data_batch3_mcs['data'], data_batch3_uws['data']], axis=0)
        data_task['label'] = np.concatenate([data_task['label'], data_batch2['label'], data_batch3_mcs['label'], data_batch3_uws['label']], axis=0)

        data_task_sub = get_data_by_labels(['Sub'], data_task)
        data_task_sub_LB, data_task_sub_RB, brain_regions = split_brain_regions(brain_regions_all, data_task_sub)
        data_task_sub = np.concatenate([data_task_sub_LB, data_task_sub_RB], axis=0)

        data_task_mcs_uws_l = get_data_by_labels(['UWS_L', 'UWS_LR', 'MCS_L', 'MCS_LR'], data_task)
        data_task_mcs_uws_l_L, _, _ = split_brain_regions(brain_regions_all, data_task_mcs_uws_l)
        data_task_mcs_uws_r = get_data_by_labels(['UWS_R', 'UWS_LR', 'MCS_R', 'MCS_LR'], data_task)
        _, data_task_mcs_uws_r_R, _ = split_brain_regions(brain_regions_all, data_task_mcs_uws_r)
        data_task_mcs_uws = np.concatenate([data_task_mcs_uws_l_L, data_task_mcs_uws_r_R], axis=0)

        data_task_mcs_l = get_data_by_labels(['MCS_L', 'MCS_LR'], data_task)
        data_task_mcs_l_L, _, _ = split_brain_regions(brain_regions_all, data_task_mcs_l)
        data_task_mcs_r = get_data_by_labels(['MCS_R', 'MCS_LR'], data_task)
        _, data_task_mcs_r_R, _ = split_brain_regions(brain_regions_all, data_task_mcs_r)
        data_task_mcs = np.concatenate([data_task_mcs_l_L, data_task_mcs_r_R], axis=0)

        data_task_uws_l = get_data_by_labels(['UWS_L', 'UWS_LR'], data_task)
        data_task_uws_l_L, _, _ = split_brain_regions(brain_regions_all, data_task_uws_l)
        data_task_uws_r = get_data_by_labels(['UWS_R', 'UWS_LR'], data_task)
        _, data_task_uws_r_R, _ = split_brain_regions(brain_regions_all, data_task_uws_r)
        data_task_uws = np.concatenate([data_task_uws_l_L, data_task_uws_r_R], axis=0)


        datas = [
            (data_task_sub, data_task_mcs_uws, brain_regions, "distinguish_between_sub_and_(uwsAndmcs)"),
            (data_task_sub, data_task_uws, brain_regions, "distinguish_between_sub_and_uws"),
            (data_task_sub, data_task_mcs, brain_regions, "distinguish_between_sub_and_mcs"),
            (data_task_mcs, data_task_uws, brain_regions, "distinguish_between_mcs_and_uws"),
        ]

        for data1, data2, brain_regions, task_name in datas:
            data1_labels = np.zeros(len(data1))
            data2_labels = np.ones(len(data2))

            X = np.vstack((data1, data2))
            Y = np.concatenate((data1_labels, data2_labels))

            best_model, feature_importance = random_forest_classification_with_gridsearch(
                data=X,
                labels=Y,
                feature_names=brain_regions,
                test_size=0.3,
                random_state=42,
                save_name=f"{save_dir}/{task_name}_{data_type}_importanceMatrix"
            )

            df = pd.DataFrame([feature_importance], columns=brain_regions)
            df.to_excel(f"{save_dir}/{task_name}_{data_type}_importanceMatrix.xlsx", index=False)

        data_sub_labels = np.zeros(len(data_task_sub))
        data_mcs_label = np.ones(len(data_task_mcs))
        data_uws_label = np.ones(len(data_task_uws)) * 2

        X = np.vstack((data_task_sub, data_task_mcs, data_task_uws))
        Y = np.concatenate((data_sub_labels, data_mcs_label, data_uws_label))

        best_model, feature_importance = random_forest_classification_with_gridsearch(
            data=X,
            labels=Y,
            feature_names=brain_regions,
            test_size=0.3,
            random_state=42,
            save_name=f"{save_dir}/three_classifications_mcs_uws_sub_{data_type}_importanceMatrix"
        )

        df = pd.DataFrame([feature_importance], columns=brain_regions)
        df.to_excel(f"{save_dir}/three_classifications_mcs_uws_sub_{data_type}_importanceMatrix.xlsx", index=False)
