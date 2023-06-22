# Copyright (c) OpenMMLab. All rights reserved.

import pickle
import json

def remove_json_anno():
    with open('data/OpenLane-V2/train/00492/info/315970269249927220.json') as f:
        data = json.load(f)
        data.pop('annotation')
        print(data)

        with open('315970269249927220.json', 'w') as file:
            json.dump(data, file)

def read_pickle():
    pf = open('work_dirs/baseline_large_res101/result.pkl', 'rb')
    data = pickle.load(pf)
    print(data['method'])
    print(data['authors'])
    print(data['e-mail'])
    print(data['institution / company'])
    print(data['country / region'])


    print(len(data['results']))

    file_out = open('work_dirs/baseline_large_res101/result_res101_e33_submitted.pkl', 'wb')
    pickle.dump(data, file_out,   protocol=pickle.DEFAULT_PROTOCOL)
    file_out.close()

    # with open('work_dirs/baseline_large/result.json', 'w') as file:
    #     json.dump(data, file)

    # for key, value in data['results'].items():
    #     print(key)
    #
    #     print(value)
    #     break

import torch
from torch.utils import benchmark
def test_torch_type(torch_type):


    typ = torch_type  # 数据精度
    n = 1024 * 16
    a = torch.randn(n, n).type(typ).cuda()
    b = torch.randn(n, n).type(typ).cuda()

    t = benchmark.Timer(
        stmt='a @ b',
        globals={'a': a, 'b': b})

    x = t.timeit(50)
    tflops = 2 * n ** 3 / x.median / 1e12
    print( f'{typ}: {tflops:.2f} tflops')


def gpu_info() -> str:
    info = ''
    for id in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(id)
        info += f'CUDA:{id} ({p.name}, {p.total_memory / (1 << 30):.2f}GB)\n'
    return info[:-1]

def gpu_tf_benchmark():
    print(gpu_info())
    type_list = [torch.float16, torch.float32]
    print(f'type_list:  {type_list}')
    for type in type_list:
        test_torch_type(type)



if __name__ == '__main__':
    # remove_json_anno()

    # read_pickle()

    gpu_tf_benchmark()

    print('done' )
