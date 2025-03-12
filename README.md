# mt_test

支持摩尔线程的8个模型和量化模型的测试

# 1. 创建 docker 容器
```shell
docker run -it --privileged --shm-size=80G  --pid=host --network=host --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g ort-musa-cmcc:v0.1-cmcc /bin/bash
```

# 2. 准备模型
所有模型放入/models中

```plain-text
/models/
├── ECAPA
│   └── voxceleb_ECAPA512.onnx
├── Retinaface
│   └── RetinaFace.onnx
├── arcface
│   └── arcfaceresnet100-8.onnx
├── fastspeech2
│   ├── fastspeech2_csmsc_am_decoder.onnx
│   ├── fastspeech2_csmsc_am_encoder_infer.onnx
│   ├── fastspeech2_csmsc_am_postnet.onnx
│   └── mb_melgan_csmsc.onnx
├── hrnet
│   └── hrnet_w18_fp32.onnx
└── resnet-q
│    └── resnet-q.onnx
├── slowfast
│   └── slowfast.onnx
└── yolov8
   └── yolov8n.onnx
```

# 3. 安装onnxruntime-musa

```shell
pip uninstall onnxruntime
pip install onnxruntime-1.18.1-cp38-cp38-linux_x86_64.whl 
```

# 4. 运行8个模型对应的测试脚本

> XPU_TYPE should be either 'musa' or 'cuda'. 

```shell
python test_arcface.py --xpu $XPU_TYPE --model /models/arcface/arcfaceresnet100-8.onnx
python test_ecapa.py --xpu $XPU_TYPE --model /models/ECAPA/voxceleb_ECAPA512.onnx
python test_hrnet.py --xpu $XPU_TYPE --model /models/hrnet/hrnet_w18_fp32.onnx
python test_retinaface.py --xpu $XPU_TYPE --model /models/Retinaface/RetinaFace.onnx
python test_slowfast.py --xpu $XPU_TYPE --model /models/slowfast/slowfast.onnx
python test_yolov8.py --xpu $XPU_TYPE --model /models/yolov8/yolov8n.onnx
python test_fastspeech2_decoder.py --xpu $XPU_TYPE --model /models/fastspeech2/fastspeech2_csmsc_am_decoder.onnx
python test_fastspeech2_encoder.py --xpu $XPU_TYPE --model /models/fastspeech2/fastspeech2_csmsc_am_encoder_infer.onnx
python test_fastspeech2_postnet.py --xpu $XPU_TYPE --model /models/fastspeech2/fastspeech2_csmsc_am_postnet.onnx
python test_mb_melgan.py --xpu $XPU_TYPE --model /models/fastspeech2/mb_melgan_csmsc.onnx
```

# 5. 运行量化resnet的测试脚本

## 安装依赖包

```shell
pip install torchvision
```
## 运行量化resnet

其中 cifar-100-python.tar.gz 会被自动下载并放在指定的 /datasets/resnet-q/ 路径下.

```shell
python test_resnet_q.py --model /models/resnet-q/resnet-q.onnx --dataset /datasets/resnet-q
```

8个网络模型预期结果如下：
| 测试脚本                  |相对误差   |
|--------------------------|-----------|
| test_arcface             |  0.0007   |
| test_ecapa               | 0.0002    |
| test_hrnet               | 0.0017    |
| test_retinaface          | 0.0068    |
| test_slowfast            | 0.0012    |
| test_yolov8              | 2.2534    |
| test_fastspeech2_decoder | 0.0064    |
| test_fastspeech2_encoder | 2.24e-05  |
| test_fastspeech2_postnet | 0.0002    |
| test_mb_melgan           | 0.412     |


量化模型预期结果如下：
| 测试脚本            | top-1         | top5          |
|--------------------|---------------|---------------|
| test_resnet_q      |   79.81%      |     95.02%     |

---

2025/03/04 测试结果：

| 测试脚本                  | 相对误差   | 每 batch 处理时间（ms） |
|--------------------------|-----------|---|
| test_arcface             | 0.0008    | 24.11 |
| test_ecapa               | 0.0002    | 7.33 |
| test_hrnet               | 0.0017    | 50.13 |
| test_retinaface          | 0.0071    | 23.31 |
| test_slowfast            | 0.0010    | 593.01 |
| test_yolov8              | 2.2404    | 13.60 |
| test_fastspeech2_decoder | 0.0056    | 1.74 |
| test_fastspeech2_encoder | 2.14e-05  | 341.51 |
| test_fastspeech2_postnet | 0.0003    | 1.77 |
| test_mb_melgan           | 0.3473    | 5.69   |

| 测试脚本            | top-1         | top5          | 每 batch 处理时间（ms） |
|--------------------|---------------|---------------|---|
| test_resnet_q      |   79.81%      |     95.02%     | 12.26 |

---

2025/03/11 A100 测试结果：

| 测试脚本                  | 相对误差   | 每 batch 处理时间（ms） |
|--------------------------|-----------|---|
| test_arcface             | 0.0004 | 4.09 |
| test_ecapa               | 0.0002 | 1.30 |
| test_hrnet               | 0.0016 | 8.82 |
| test_retinaface          | 0.0068 | 5.05 |
| test_slowfast            | 0.0012 | 11.38 |
| test_yolov8              | 0.0030 | 3.98 |
| test_fastspeech2_decoder | 0.0043 | 0.23 |
| test_fastspeech2_encoder | 2.15e-05 | 36.58 |
| test_fastspeech2_postnet | 0.0003 | 0.21 |
| test_mb_melgan           | 0.3987 | 0.97 |
