import os

import numpy as np
import pandas as pd

from utils import load_processed_data, get_data_by_labels, split_brain_regions


if __name__ == '__main__':
    save_dir = './results/task_state'
    os.makedirs(save_dir, exist_ok=True)
    data_name = load_processed_data('data/task_state/name.npy')
    data_binary = load_processed_data('data/task_state/yesOrNo.npy')
    data_imagine = load_processed_data('data/task_state/imagination.npy')

    data_3tasks = {
        "data": data_name["data"].copy(),
        "label": data_name["label"].copy(),
        "label2id": data_name["label2id"],
        "id2label": data_name["id2label"],
    }
    for data in [data_binary, data_imagine, ]:
        data_3tasks["data"] = np.concatenate([data_3tasks["data"], data["data"]], axis=0)
        data_3tasks["label"] = np.concatenate([data_3tasks["label"], data["label"]], axis=0)

    df = pd.read_excel('data/task_state/brain_region.xlsx', sheet_name='Sheet1', header=0)
    brain_regions_all = df.iloc[:, 0].tolist()

    for task_name, task_data in [('task_name', data_name), ('task_yesOrNo', data_binary), ('task_imagination', data_imagine), ('task_all', data_3tasks)]:
        data_task_sub = get_data_by_labels(['Sub'], task_data)
        data_task_sub_LB, data_task_sub_RB, brain_regions = split_brain_regions(brain_regions_all, data_task_sub)
        data_task_sub = np.concatenate([data_task_sub_LB, data_task_sub_RB], axis=0)

        data_task_uws_l = get_data_by_labels(['UWS_L', 'UWS_LR'], task_data)
        data_task_uws_l_L, _, _ = split_brain_regions(brain_regions_all, data_task_uws_l)
        data_task_uws_r = get_data_by_labels(['UWS_R', 'UWS_LR'], task_data)
        _, data_task_uws_r_R, _ = split_brain_regions(brain_regions_all, data_task_uws_r)
        data_task_uws = np.concatenate([data_task_uws_l_L, data_task_uws_r_R], axis=0)

        datas = [
            (data_task_sub, data_task_uws, brain_regions),
        ]

        for data1, data2, brain_regions in datas:
            data1_labels = np.zeros(len(data1))
            data2_labels = np.ones(len(data2))

            # X = np.vstack((data1, data2))
            # Y = np.concatenate((data1_labels, data2_labels))
            #
            # best_model, feature_importance = random_forest_classification_with_gridsearch(
            #     data=X,
            #     labels=Y,
            #     feature_names=brain_regions,
            #     test_size=0.3,
            #     random_state=42,
            #     save_name=f'{save_dir}/{task_name}_importanceMatrix'
            # )

            # df = pd.DataFrame([feature_importance], columns=brain_regions)
            # df.to_excel(f"{save_dir}/{task_name}_importanceMatrix.xlsx", index=False)