# mt_test

支持摩尔线程的8个模型和量化模型的测试

```shell
# 1. 创建docker 容器
docker run -it --privileged --shm-size=80G  --pid=host --network=host --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g ort-musa-cmcc:v0.1-cmcc /bin/bash

# 2. 准备模型
所有模型放入/models中

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


# 3. 安装onnxruntime-musa
pip uninstall onnxruntime
pip install onnxruntime-1.18.1-cp38-cp38-linux_x86_64.whl 

# 4. 运行8个模型对应的测试脚本
python test_arcface.py /models/arcface/arcfaceresnet100-8.onnx
python test_ecapa.py /models/ECAPA/voxceleb_ECAPA512.onnx
python test_hrnet.py /models/hrnet/hrnet_w18_fp32.onnx
python test_retinaface.py /models/Retinaface/RetinaFace.onnx
python test_slowfast.py /models/slowfast/slowfast.onnx
python test_yolov8.py /models/yolov8/yolov8n.onnx
python test_fastspeech2_encoder.py /models/fastspeech2/fastspeech2_csmsc_am_encoder_infer.onnx
python test_fastspeech2_decoder.py /models/fastspeech2/fastspeech2_csmsc_am_decoder.onnx
python test_fastspeech2_postnet.py /models/fastspeech2/fastspeech2_csmsc_am_postnet.onnx
python test_mb_melgan.py /models/fastspeech2/mb_melgan_csmsc.onnx

# 5. 运行量化resnet的测试脚本
# 安装依赖包
pip install torchvision
# 运行量化resnet，其中cifar-100-python.tar.gz会被自动下载并放在指定的/datasets/resnet-q/路径下
python test_resnet_q.py --model /models/resnet-q/resnet-q.onnx --dataset /datasets/resnet-q
```

网络模型预期结果如下：
| 测试脚本                  | 最大差值  | 相对误差  |
|--------------------------|----------|-----------|
| test_arcface             | 0.00314  | 0.0007    |
| test_ecapa               | 0.00091  | 0.0002    |
| test_hrnet               | 0.01099  | 0.0017    |
| test_retinaface          | 0.09877  | 0.0068    |
| test_slowfast            | 0.00733  | 0.0012    |
| test_yolov8              | 617.333  | 2.2534    |
| test_fastspeech2_decoder | 0.07310  | 0.0064    |
| test_fastspeech2_encoder | 0.00051  | 2.24e-05  |
| test_fastspeech2_postnet | 0.00175  | 0.0002    |
| test_mb_melgan           | 1.56622  | 0.412     |


量化模型预期结果如下：
| 测试脚本            | top-1         | top5          |
|--------------------|---------------|---------------|
| test_resnet_q      |   79.81%      |     95.02%     |