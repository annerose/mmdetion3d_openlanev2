import os

import shutil
import numpy as np
import json
import pickle


def reduce_raw_result_by_confidence(raw_result_list):

    lc_confidence_max = 0.3
    te_confidence_max = 0.3

    filter_raw_result_list = []
    for frame in raw_result_list:
        # linecenter
        lc = frame['pred_lc'][0]
        lc_conf = frame['pred_lc'][1][:, 0]

        lc_index = np.where(lc_conf > lc_confidence_max)
        lc_filter = lc[lc_index]
        lc_conf_filter = np.expand_dims(lc_conf[lc_index], 1)

        # traffic elem
        te = frame['pred_te'][0]
        te_conf = frame['pred_te'][1]

        te_index = np.where(te_conf > te_confidence_max)
        te_filter = te[te_index]
        te_conf_filter = te_conf[te_index]

        # topo
        lclc = frame['pred_topology_lclc']
        lclc_filter = np.squeeze(lclc[lc_index][:, lc_index], axis=1)

        lcte = frame['pred_topology_lcte']
        lcte_filter = np.squeeze(lcte[lc_index][:, te_index], axis=1)

        # print(lclc_filter.shape, lcte_filter.shape)

        frame_filter = {
            'pred_lc': [lc_filter, lc_conf_filter],
            'pred_te': [te_filter, te_conf_filter],
            'pred_topology_lclc': lclc_filter,
            'pred_topology_lcte': lcte_filter
        }

        filter_raw_result_list.append(frame_filter)

    return filter_raw_result_list


def  read_pickle():

    with open('work_dirs/baseline_large/raw_result.pkl', 'rb') as f:
        data = pickle.load(f)

        reduce_raw_result_by_confidence(data)






        print('111' )

        # print(len(data['results']))
    # for key, value in data['results'].items():
    #     # print(key)
    #     predictions = value['predictions']
    #     lane_centerline = predictions['lane_centerline']
    #     # for cl in lane_centerline:
    #     #     print(cl['confidence'])
    #
    #     # traffic_element = predictions['traffic_element']
    #     # for te in traffic_element:
    #     #     print(te['confidence'])
    #
    #     print(predictions['topology_lclc'].shape, predictions['topology_lcte'].shape)
    #
    #     # break





def read_json():
    json_file = r'F:\work\code\mmdetection3d-1.0.0rc6\data\OpenLane-V2\test\00556\info\315969900049927216.json'

    with open(json_file) as f:
        data = json.load(f)
        data.pop('annotation')

    with open('315969900049927216.json', 'w') as f2:
        json.dump(data, f2)



import torch.nn.functional as functional
import torch

def linear(x, weight, bias=None):
    if bias is None:
        return torch.matmul(x, weight.T)
    else:
        return torch.matmul(x, weight.T) + bias


def test_linear():

    x = torch.rand(2, 8)
    w = torch.rand(2, 8)

    b = torch.rand(2, 2)
    # torch.matmul(x, w.T) + b
    result = linear(x, w, b)

    result2 = functional.linear(x, w, b)
    loss = torch.sum(result - result2)
    print(f'result: {result}')

    print(f'result2: {result2}')
    print(f'loss: {loss}')




if __name__ == '__main__':
    # read_pickle()

    test_linear()
    print('done')