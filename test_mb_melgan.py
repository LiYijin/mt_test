import torch
import onnxruntime as ort
import numpy as np
import random
import sys
import time

import argparse

ort_type_to_numpy_type_map = {
            "tensor(int64)": np.longlong,
            "tensor(int32)": np.intc,
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(bool)": bool,
        }


def evaluate_fastspeech2_decoder(model_path : str, xpu_type: str):
    if xpu_type == 'musa': 
        test_xpu_session = ort.InferenceSession(model_path, providers=['MUSAExecutionProvider'])
    else:
        test_xpu_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    test_cpu_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    inputs_info = test_cpu_session.get_inputs()
    outputs_info = test_cpu_session.get_outputs()

    output_names = []

    input_dict = {}
    input_data = np.random.randn(1, 80).astype(np.float32)
    input_dict["logmel"] = input_data

    # for input in inputs_info:
    #     print(input.shape)
    #     print(input.type)
    #     input_data = np.random.randn(*input.shape).astype(ort_type_to_numpy_type_map[input.type])
    #     input_dict[input.name] = input_data

    for output in outputs_info:
        output_names.append(output.name)

    warm_up = 10
    iter = 10


    cpu_reslut = test_cpu_session.run(output_names, input_dict)
    xpu_result = []
    total_time = 0.0

    for i in range(warm_up):
        test_xpu_session.run(output_names, input_dict)

    for i in range(iter):
        start_time = time.time()
        xpu_result = test_xpu_session.run(output_names, input_dict)
        total_time += time.time() - start_time
    

    max_difference = 0.0
    L2norm = 0.0

    max_difference = np.max(np.abs(xpu_result[0] - cpu_reslut[0]))
    L2norm = np.sum(np.abs(xpu_result[0] - cpu_reslut[0])) / xpu_result[0].size

    print("Batch Size: {}\nTotal Time: {:.2f} Seconds\nLatency: {:.2f} ms / batch".format(iter, total_time, 1000.0 * total_time / iter))

    return max_difference, L2norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--model', type=str, help='Specify model path of current network. ', required=True) 
    parser.add_argument('--xpu', type=str, default='xpu', help='Specify xpu type of current test task. The default device is CUDA. ') 
    args = parser.parse_args() 
    
    md, l2 = evaluate_fastspeech2_decoder(args.model, args.xpu)
    
    print("Max: ", md)
    print("Relative Difference: ", l2)