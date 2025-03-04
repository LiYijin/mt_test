import onnxruntime as ort
import numpy as np
import random
import sys
import time

ort_type_to_numpy_type_map = {
            "tensor(int64)": np.longlong,
            "tensor(int32)": np.intc,
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(bool)": bool,
        }


def evaluate_fastspeech2_decoder(model_path : str):

    test_musa_session = ort.InferenceSession(model_path, providers=['MUSAExecutionProvider'])
    test_cpu_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    inputs_info = test_cpu_session.get_inputs()
    outputs_info = test_cpu_session.get_outputs()

    output_names = []

    input_dict = {}
    input_data = np.random.randn(1, 100, 384).astype(np.float32)
    input_dict["xs"] = input_data

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
    musa_result = []
    total_time = 0.0

    for i in range(warm_up):
        test_musa_session.run(output_names, input_dict)

    for i in range(iter):
        start_time = time.time()
        musa_result = test_musa_session.run(output_names, input_dict)
        total_time += time.time() - start_time


    max_difference = 0.0
    L2norm = 0.0

    max_difference = np.max(np.abs(musa_result[0] - cpu_reslut[0]))
    L2norm = np.sum(np.abs(musa_result[0] - cpu_reslut[0])) / cpu_reslut[0].size
    
    print("Batch Size: {}\nTotal Time: {:.2f} Seconds\nLatency: {:.2f} ms / batch".format(iter, total_time, 1000.0 * total_time / iter))

    return max_difference, L2norm

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_script.py <path_to_onnx_model>")
        sys.exit(1)
    
    md, l2 = evaluate_fastspeech2_decoder(sys.argv[1])
    
    print("Max: ", md)
    print("Relative Difference: ", l2)