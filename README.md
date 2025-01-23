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
| 测试脚本                  | 最大差值      | 相对误差                  |
|--------------------------|------------------------------------------|
| test_arcface             | 0.003143132  | 0.0007741070003248751     |
| test_ecapa               | 0.000914872  | 0.00022674076414356628    |
| test_hrnet               | 0.01099968   | 0.001766887903213501      |
| test_retinaface          | 0.09877767   | 0.006807587033226377      |
| test_slowfast            | 0.007330895  | 0.0011986882239580154     |
| test_yolov8              |  617.33344   | 2.253497909580499         |
| test_fastspeech2_decoder | 0.07310486   | 0.006474185943603516      |
| test_fastspeech2_encoder | 0.000510931  | 2.24266500061724e-05      |
| test_fastspeech2_postnet | 0.001757681  | 0.00028014618158340456    |
| test_mb_melgan           | 1.5662293    | 0.41269554138183595       |


量化模型预期结果如下：
| 测试脚本            | top-1         | top5          |
|--------------------|---------------|---------------|
| test_resnet_q      |   79.81%      |     95.02%     |